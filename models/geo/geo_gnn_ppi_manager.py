import os.path as osp

import torch
from sklearn import metrics
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI
import numpy as np

from models.geo.geo_gnn import GraphNet
from models.gnn_manager import GNNManager
from models.model_utils import TopAverage, process_action
from models.controller.multi_controller import state_space

def load_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader

def standarizing_features(train_loader,):
    from sklearn.preprocessing import StandardScaler
    train_feats = []
    for data in train_loader:
        train_feats.append(data.x.numpy())
    train_feats = np.concatenate(train_feats)
    scaler = StandardScaler()
    scaler.fit(train_feats)
    # features_ = scaler.transform(features_)
    return scaler

class GeoPPIManager(GNNManager):
    def __init__(self, args):
        super(GeoPPIManager, self).__init__(args)

        self.train_loader, self.val_loader, self.test_loader = load_data()
        self.StandardScaler = standarizing_features(self.train_loader)
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')

        self.reward_manager = TopAverage(10)
        self.args = args
        self.in_feats = args.in_feats
        self.n_classes = args.num_class
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.param_file = args.param_file
        self.shared_params = None
        self.state_space = state_space
        self.load_param()

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=True,
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

    def train(self, actions, retrain_epoch=None, from_scratch=False):

        actions = process_action(actions, self.args)
        model = self.build_gnn(actions)
        num_params = sum([param.nelement() for param in model.parameters()])
        print("model parameter num ------------->", num_params)
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
        loss_op = torch.nn.BCEWithLogitsLoss()
        if retrain_epoch is None:
            maximum_epoch = self.epochs
        else:
            maximum_epoch = retrain_epoch
        f1_vals = []
        f1_tests = []
        try:
            for epoch in range(1, maximum_epoch + 1):
                model.train()
                total_loss = self.run_model(model, optimizer, loss_op)
                f1_val = self.test(model, self.val_loader)
                f1_test = self.test(model, self.test_loader)
                f1_vals.append(f1_val)
                f1_tests.append(f1_test)
                print('Epoch: {:02d}, Loss: {:.4f}, val_f1: {:.4f}, test_f1:{:.4f}'.format(epoch, total_loss, f1_val,
                                                                                             f1_test))
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e
        # run on val data
        f1_val = self.test(model, self.val_loader)
        reward = self.reward_manager.get_reward(f1_val)
        # if the reward is better, then keep parameters
        self.save_param(model, update_all=(reward > 0))
        if len(f1_vals) > 0 and len(f1_tests) > 0:
            max_idx = np.argmax(f1_vals)
            max_f1Val = f1_vals[max_idx]
            max_f1Test = f1_tests[max_idx]
        else:
            max_f1Val = 0
            max_f1Test = 0
        return reward, f1_val, max_f1Val, max_f1Test, num_params

    def run_model(self, model, optimizer, loss_op):
        model.train()
        total_loss = 0
        for data in self.train_loader:
            num_graphs = data.num_graphs
            data.batch = None
            # data.x = torch.tensor(self.StandardScaler.transform(data.x.numpy()), dtype=torch.float)
            # print(data.x, np.max(data.x.numpy()), np.min(data.x.numpy()))
            data = data.to(self.device)
            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index), data.y)
            total_loss += loss.item() * num_graphs
            loss.backward()
            if self.args.child_model_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), self.args.child_model_grad_clip)
            optimizer.step()
        return total_loss / len(self.train_loader.dataset)

    def test(self, model, loader):
        model.eval()

        total_micro_f1 = 0
        for data in loader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                # data.x = torch.tensor(self.StandardScaler.transform(data.x.numpy()), dtype=torch.float)
                # print(data.x)
                out = model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = (out > 0).float().cpu()
            micro_f1 = metrics.f1_score(data.y, pred, average='micro')
            total_micro_f1 += micro_f1 * data.num_graphs
        return total_micro_f1 / len(loader.dataset)

    # evaluate model from scratch
    def retrain(self, actions, epochs=None):
        return self.train(actions, retrain_epoch=epochs, from_scratch=True)

    # evaluate model with a few training or no training
    def test_with_param(self, actions, retrain_epoch=None):
        try:
            return self.train(actions, retrain_epoch=retrain_epoch)
        except RuntimeError as e:
            if "CUDA" in str(e):
                return None
            else:
                raise e

