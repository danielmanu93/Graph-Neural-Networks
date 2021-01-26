import glob
import os

import numpy as np
import scipy.signal
import torch.nn.parallel
from torch import nn
import operator

import models
import utils
from models.controller.multi_controller import MultiController
from models.geo.geo_gnn_citation_manager import GeoCitationManagerManager
from models.geo.geo_gnn_ppi_manager import GeoPPIManager
from models.controller.multi_controller import component_space_skip
logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim



class Trainer(object):
    """Manage the training process"""

    def __init__(self, args):
        """
        Constructor for training algorithm.
        Build sub-model(shared) and controller.
        Build optimizer and cross entropy loss for controller.
        Optimizer and loss function managed by sub-model.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.epoch = 0
        self.start_epoch = 0
        self.cuda = args.cuda
        self.search_record = args.search_record
        self.sample_record = args.sample_record

        # variable for multi-component controller
        self.init_actions = []
        self.init_f1 = []
        self.best_actions = []
        self.best_f1 = []
        self.baseline = {}
        for com in component_space_skip.values():
            self.baseline[com] = None

        # remove regulation
        self.shared = None
        self.controller = None
        self.build_model()  # build controller and sub-model
        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)


        self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        if self.args.search_mode == "nas":
            self.args.share_param = False
            self.with_retrain = True
            self.retrain_epoch = self.args.retrain_epochs
            logger.info("NAS-like mode: retrain without share param")
            pass
        else:
            self.args.share_param = True
            self.with_retrain = False
            self.retrain_epoch = self.args.noRetrain_epochs
            logger.info("Shared parameter mode: no retrain without share param")
            pass

        if self.args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            self.shared = GeoCitationManagerManager(self.args)
        elif self.args.dataset == "PPI":
            self.shared = GeoPPIManager(self.args)
        else:
            raise Exception(f'The dataset of {self.args.dataset} has not been included')

        if self.args.controller_mode == "multi":
            self.controller = MultiController(self.args, cuda=self.args.cuda, skip_conn=self.args.residual,
                                              num_layers=self.args.layers_of_child_model,
                                              num_classes=self.args.num_class, num_selectCom=self.args.num_selectCom)
        else:
            raise Exception(f'The controller of {self.args.controller_mode} has not been included')

        if self.cuda:
            self.controller.cuda()

    def multi_init_actions(self, batch_size):
        print("Initiate architectures in the multi-component search method")
        init_actions = self.controller.init_actions(batch_size)
        for action_i, actions in enumerate(init_actions):
            results = self.shared.test_with_param(actions, self.retrain_epoch)
            if results is None:  # the model is oversized
                f1_val = 0
            else:
                f1_val = results[1]
            self.update_action(actions, action_i, f1_val)
            self.update_best_action(actions, f1_val)

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained for exploration.
        - In the second phase, the controller's parameters are trained.
        """
        self.multi_init_actions(self.args.controller_max_step)

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                self.save_model()

        self.evaluate(self.best_actions)
        # self.save_model()

    def update_action(self, action, action_i, f1_val):
        # update the architectures in self.init_actions and self.best_actions
        if len(self.init_f1) < self.args.controller_max_step:
            self.init_actions.append(action)
            self.init_f1.append(f1_val)
        else:
            if f1_val > self.init_f1[action_i]:
                self.init_actions[action_i] = action
                self.init_f1[action_i] = f1_val
            else:
                pass

    def update_best_action(self, action, f1_val):
        # check whether there is already the same architecture
        eq_flag = False
        eq_index = 100000
        for i in range(len(self.best_actions)):
            eq_flag = operator.eq(action, self.best_actions[i])
            if eq_flag:
                eq_index = i
                break

        # update the best architecture
        if eq_flag:
            if f1_val > self.best_f1[eq_index]:
                self.best_f1[eq_index] = f1_val
            else:
                pass
        else:
            if len(self.best_actions) < self.args.topK_actions:
                self.best_f1.append(f1_val)
                self.best_actions.append(action)
            else:
                min_idx = np.argmin(self.best_f1)
                if f1_val > self.best_f1[min_idx]:
                    self.best_f1[min_idx] = f1_val
                    self.best_actions[min_idx] = action


    def get_reward(self, action, old_f1=None):
        """
        Computes the reward of a single sampled model on validation data.
        """

        results = self.shared.test_with_param(action, self.retrain_epoch)
        if results is None:  # the model is oversized
            reward = self.args.penalty_oversize
            f1_val = 0
        else:
            if old_f1 is not None:
                f1_val = results[1]
                reward = np.clip(f1_val-old_f1, -0.5, 0.5)
            else:
                reward = results[0]
                f1_val = results[1]
        return reward, f1_val

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        for step in range(self.args.controller_max_step):
            # sample models
            structure_list, log_probs, entropies, components = self.controller.sample(self.init_actions[step],
                                                                                      with_details=True)
            structure_list = structure_list[0]
            log_probs = log_probs[0]
            entropies = entropies[0]
            components = components[0]
            print(components)
            reward, f1_val = self.get_reward(structure_list, old_f1=self.init_f1[step])
            # update actions
            self.update_action(structure_list, step, f1_val)
            self.update_best_action(structure_list, f1_val)
            # record the f1 score of searched architecture
            with open(self.search_record, "a") as f:
                msg = f"actions:{structure_list}, component:{components}, f1_val:{f1_val}\n"
                f.write(msg)

            total_loss = None
            for i in range(self.args.num_selectCom):
                # compute rewards
                np_entropies = entropies[i].data.cpu().numpy()
                # print(reward/2, np.sum(np_entropies))
                if self.args.entropy_mode == 'reward':
                    rewards = reward/self.args.num_selectCom + self.args.entropy_coeff * np_entropies
                elif self.args.entropy_mode == 'regularizer':
                    rewards = reward/self.args.num_selectCom * np.ones_like(np_entropies)
                else:
                    raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')


                torch.cuda.empty_cache()
                # discount
                if 1 > self.args.discount > 0:
                    rewards = discount(rewards, self.args.discount)
                # moving average baseline
                baseline = self.baseline[components[i]]
                if baseline is None:
                    baseline = rewards
                else:
                    decay = self.args.ema_baseline_decay
                    baseline = decay * baseline + (1 - decay) * rewards
                self.baseline[components[i]] = baseline
                adv = rewards - baseline
                adv = utils.get_variable(adv, self.cuda, requires_grad=False)
                # policy loss
                loss = -log_probs[i] * adv
                if self.args.entropy_mode == 'regularizer':
                    loss -= self.args.entropy_coeff * np_entropies
                loss = loss.sum()  # or loss.mean()
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss

            # update
            self.controller_optim.zero_grad()
            total_loss.backward()
            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()
            self.controller_step += 1
            torch.cuda.empty_cache()

        print("*" * 35, "training controller over", "*" * 35)

    def evaluate(self, best_actions):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        for actions in best_actions:
            results = self.shared.retrain(actions, self.args.fromScratch_epochs)
            if results is not None:
                f1_val = results[2]
                f1_test = results[3]
                num_params = results[4]
                with open(self.sample_record, "a") as f:
                    msg = f"best actions:{actions}, params: {num_params}, f1_val:{f1_val}, f1_test:{f1_test}\n"
                    f.write(msg)
                    print(msg)
            else:
                continue
        print('use the architecture with the best validation accuracy')


    @property
    def controller_path(self):
        return f'params/{self.args.dataset}/{self.args.controller_mode}/' \
               f'{self.args.search_mode}_controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'params/{self.args.dataset}/{self.args.controller_mode}/' \
               f'{self.args.search_mode}_controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join('params', self.args.dataset, self.args.controller_mode,
                                       f'{self.args.search_mode}_controller*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 2, 'epoch', 'controller')
        controller_steps = get_numbers(basenames, '_', 3, 'step', 'controller')

        epochs.sort()
        controller_steps.sort()

        return epochs, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join('params', self.args.dataset, self.args.controller_mode,
                             f'{self.args.search_mode}_controller_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.args.dataset}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        logger.info(f'[*] LOADED: {self.controller_path}')
