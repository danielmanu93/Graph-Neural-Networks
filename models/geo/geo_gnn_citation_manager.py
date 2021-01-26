import heapq
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.geo.geo_gnn import GraphNet
from models.gnn_manager import GNNManager
from models.gnn_citation_manager import evaluate
from models.model_utils import TopAverage, process_action
from models.controller.multi_controller import state_space
from models.model_utils import EarlyStop
from torch_geometric.data import DataLoader


def load_data(dataset="Cora"):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())

    return dataset[0]


class GeoCitationManagerManager(GNNManager):
    def __init__(self, args):
        super(GeoCitationManagerManager, self).__init__(args)

        self.data = load_data(args.dataset)
        self.args.in_feats = self.in_feats = self.data.num_features
        self.args.num_class = self.n_classes = self.data.y.max().item() + 1
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.data.to(self.device)

        self.reward_manager = TopAverage(10)
        self.args = args
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10
        self.param_file = args.param_file
        self.state_space = state_space
        self.loss_fn = torch.nn.functional.nll_loss
        self.shared_params = None
        self.load_param()

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=self.args.batch_normal, state_num=len(self.state_space),
                         residual=self.args.residual, layer_nums=self.args.layers_of_child_model,
                         dataset=self.args.dataset)
        return model

    def load_param(self):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:  # not share parameters
                return
        if osp.exists(self.param_file):
            self.shared_params = torch.load(self.param_file)

    def save_param(self, model, update_all=False):
        if hasattr(self.args, "share_param"):
            if not self.args.share_param:
                return
        model.cpu()
        if isinstance(model, GraphNet):
            self.shared_params = model.get_param_dict(self.shared_params, update_all)
            torch.save(self.shared_params, self.param_file)
        model.to(self.device)

    def train(self, actions, retrain_epoch=None, from_scratch=False):

        actions = process_action(actions, self.args)
        model = self.build_gnn(actions)
        print("train action:", actions)

        # eval number of params of model, big model will be drop
        num_params = sum([param.nelement() for param in model.parameters()])
        if num_params > self.args.max_param:
            print(f"model too big, num_param more than {self.args.max_param}")
            del model
            return None

        # share params
        if not from_scratch:
            model.load_param(self.shared_params)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if retrain_epoch is None:
            maximum_epoch = self.epochs
        else:
            maximum_epoch = retrain_epoch
        acc_vals = []
        best_test_performance = 0
        best_val_performance = 0
        early_stop_manager = EarlyStop(100)
        history_avgAcc = self.reward_manager.return_average()
        try:
            for epoch in range(1, maximum_epoch + 1):
                total_loss = self.run_model(model, optimizer, self.loss_fn, self.data)
                acc_train, acc_val, loss_val, acc_test = self.test(model, self.data, self.loss_fn)
                #if acc_test > 0.8:
                print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, total_loss,
                                                                                                 acc_val, acc_test))

                acc_vals.append(acc_val)
                if early_stop_manager.should_save(total_loss, acc_train, loss_val, acc_val):
                    if acc_val > history_avgAcc:
                        self.save_param(model, update_all=True)
                    else:
                        self.save_param(model, update_all=False)
                    best_test_performance = acc_test
                    best_val_performance = acc_val
                if early_stop_manager.should_stop(total_loss, acc_train, loss_val, acc_val):
                    print("early stop")
                    break
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e

        if len(acc_vals) == 0:
            acc_val = 0
        else:
            acc_val = np.mean(heapq.nlargest(5, acc_vals))
        reward = self.reward_manager.get_reward(acc_val)
        return reward, acc_val, best_val_performance, best_test_performance

    def run_model(self, model, optimizer, loss_fn, data):
        model.train()
        # torch.cuda.empty_cache()
        # forward
        batch_size = 20
        tr_step = 0
        loss_step = 0
        tr_size = data.x.size()[0]
        total_loss = 0
        while tr_step * batch_size < tr_size:
            if data.train_mask[tr_step*batch_size:(tr_step+1)*batch_size].sum().item() > 0:
                mask = torch.tensor([0]*tr_size, dtype=torch.uint8)
                mask[tr_step*batch_size:(tr_step+1)*batch_size] = data.train_mask[tr_step*batch_size:(tr_step+1)*batch_size]
                mask = mask.to(self.device)
                logits = model(data.x, data.edge_index)
                logits = F.log_softmax(logits[mask], 1)
                loss = loss_fn(logits, data.y[mask])
                optimizer.zero_grad()
                loss.backward()
                if self.args.child_model_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), self.args.child_model_grad_clip)
                optimizer.step()
                total_loss += loss.item()
                loss_step += 1
            else:
                pass
            tr_step += 1
        return total_loss / loss_step

        # logits = model(data.x, data.edge_index)
        # logits = F.log_softmax(logits, 1)
        # loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        # optimizer.zero_grad()
        # loss.backward()
        # if self.args.child_model_grad_clip > 0:
        #     torch.nn.utils.clip_grad_norm(model.parameters(), self.args.child_model_grad_clip)
        # optimizer.step()
        # return loss

    def test(self, model, data, loss_fn):
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        logits = F.log_softmax(logits, 1)
        acc_train = evaluate(logits, data.y, data.train_mask)
        acc_valid = evaluate(logits, data.y, data.val_mask)
        acc_test = evaluate(logits, data.y, data.test_mask)
        loss_val = float(loss_fn(logits[data.val_mask], data.y[data.val_mask]))

        return acc_train, acc_valid, loss_val, acc_test

    def retrain(self, actions, epochs=None):
        actions = process_action(actions, self.args)
        model = self.build_gnn(actions)
        num_params = sum([param.nelement() for param in model.parameters()])
        print("train the best action:", actions)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_trains = []
        acc_vals = []
        acc_tests = []
        early_stop_manager = EarlyStop(100)
        best_test_performance = 0
        best_valid_performance = 0
        lr = self.lr
        try:
            for epoch in range(1, epochs + 1):
                # if epoch%1000 == 0 and epoch > 0:
                #     lr = lr/2
                #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
                total_loss = self.run_model(model, optimizer, self.loss_fn, self.data)
                acc_train, acc_val, loss_val, acc_test = self.test(model, self.data, self.loss_fn)
                print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, total_loss,
                                                                                             acc_val, acc_test))
                loss_trains.append(total_loss)
                acc_vals.append(acc_val)
                acc_tests.append(acc_test)

                if early_stop_manager.should_save(total_loss, acc_train, loss_val, acc_val):
                    best_valid_performance = acc_val
                    best_test_performance = acc_test

                if early_stop_manager.should_stop(total_loss, acc_train, loss_val, acc_val):
                    print("early stop")
                    break
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e

        max_idxs = np.argsort(acc_vals)[-10:]  # 10 for Cora, Citeseer or 20 for Pubmed
        max_loss = 0
        max_idx = 0
        for i in range(len(max_idxs)):
            if loss_trains[max_idxs[i]] > max_loss:
                max_loss = loss_trains[max_idxs[i]]
                max_idx = max_idxs[i]
        best_valid_performance = acc_vals[max_idx]
        best_test_performance = acc_tests[max_idx]

        return 0, 0, best_valid_performance, best_test_performance, num_params

    # evaluate model with a few training or no training
    def test_with_param(self, actions, retrain_epoch=None):
        try:
            return self.train(actions, retrain_epoch=retrain_epoch)
        except RuntimeError as e:
            if "CUDA" in str(e):
                return None
            else:
                raise e