"""A module with NAS controller-related code."""

import torch
import torch.nn.functional as F
import numpy as np
import utils
from torch.autograd import Variable
import random

from models.controller.multi_controller import state_space

component_space_skip = {0:"skip", 1: "attention_type", 2:"aggregator_type",
                   3:"activate_function", 4:"number_of_heads", 5:"update_type", 6:"hidden_units"}
component_space = {0: "attention_type", 1:"aggregator_type", 2:"activate_function",
                   3:"number_of_heads", 4:"update_type", 5:"hidden_units"}

# the output of controller RNN is index, translate it into operator name
def _construct_action(actions, state_space):
    state_length = len(state_space)
    keys = list(state_space.keys())
    layers = []
    for action in actions:
        predicted_actions = []
        for index, each in enumerate(action):
            state_index = index % state_length
            predicted_actions.append(state_space[keys[state_index]][each])
        layers.append(predicted_actions)
    return layers

# Skip connection: the output of controller RNN is index, translate it into operator name
def _construct_action_skip(actions, state_space, num_layers):
    state_length = len(state_space)
    keys = list(state_space.keys())
    layers = []
    for action in actions:
        predicted_actions = []
        start_index = 0
        for i in range(1, num_layers+1):
            for index in range(start_index, start_index+state_length+i):
                if index-start_index < i:
                    predicted_actions.append(action[index])
                else:
                    state_index = (index-start_index-i) % state_length
                    predicted_actions.append(state_space[keys[state_index]][action[index]])
            start_index += state_length+i
        layers.append(predicted_actions)
    return layers

def _deconstruct_action(actions, state_space, num_tokens):
    state_length = len(state_space)
    keys = list(state_space.keys())
    layers = []
    for action in actions:
        predicted_actions = []
        for index, each in enumerate(action):
            state_index = index % state_length
            if index == len(action)-1:
                # delete the last action in architecture, which is the output dimension
                continue
            state_position = state_space[keys[state_index]].index(each)
            embedding_position = int(sum(num_tokens[:index]) + state_position)
            predicted_actions.append(embedding_position)
        layers.append(predicted_actions)
    return layers

def _deconstruct_action_skip(actions, state_space, num_tokens, num_layers):
    state_length = len(state_space)
    keys = list(state_space.keys())
    layers = []
    for action in actions:
        predicted_actions = []
        start_index = 0
        for i in range(1, num_layers + 1):
            for index in range(start_index, start_index + state_length + i):
                if index - start_index < i:
                    skip_index = index - start_index
                    token_index = (i-1) * (state_length+1)
                    embedding_position = sum(num_tokens[:token_index]) + skip_index*2 + action[index]
                else:
                    state_index = (index - start_index - i) % state_length
                    if index == len(action)-1:
                        # delete the last action in architecture, which is the output dimension
                        continue
                    state_position = state_space[keys[state_index]].index(action[index])
                    token_index = (i-1) * (state_length+1) + state_index + 1
                    embedding_position = sum(num_tokens[:token_index]) + state_position
                predicted_actions.append(embedding_position)
            start_index += state_length + i
        layers.append(predicted_actions)
    return layers

def _replace_action(best_action, actions, pop_index, component, state_space):
    layers = []
    for action in actions:
        predicted_action = best_action
        for i in range(len(pop_index)):
            if component == "skip":
                predicted_action[pop_index[i]] = action[i]
            else:
                predicted_action[pop_index[i]] = state_space[component][action[i]]
        layers.append(predicted_action)
    return layers


class MultiController(torch.nn.Module):

    def __init__(self, args, num_layers=3, skip_conn=False, controller_hid=100, cuda=True, mode="train",
                 softmax_temperature=5.0, tanh_c=2.5, num_classes=121):
        torch.nn.Module.__init__(self)
        self.mode = mode
        self.num_layers = num_layers
        self.skip_conn = skip_conn
        self.controller_hid = controller_hid
        self.num_classes = num_classes
        self.is_cuda = cuda
        self.dataset = args.dataset
        if args and args.softmax_temperature:
            self.softmax_temperature = args.softmax_temperature
        else:
            self.softmax_temperature = softmax_temperature
        if args and args.tanh_c:
            self.tanh_c = args.tanh_c
        else:
            self.tanh_c = tanh_c

        state_space_length = []
        keys = state_space.keys()
        for key in keys:
            state_space_length.append(len(state_space[key]))
        self.num_tokens = []
        if not skip_conn:
            for _ in range(self.num_layers):
                self.num_tokens += state_space_length
        else:
            for idx in range(1, self.num_layers + 1):
                self.num_tokens += [idx*2]  # one for skip connection, and the other one for empty
                self.num_tokens += state_space_length
        self.num_tokens.pop()  # delete the output dimension at the last layer, and replace it with self.num_classes
        num_total_tokens = sum(self.num_tokens)
        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        self.decoders = []
        if not skip_conn:
            # share the same decoder
            for idx, size in enumerate(state_space_length):
                decoder = torch.nn.Linear(controller_hid, size)
                self.decoders.append(decoder)
        else:
            for idx in range(1, self.num_layers + 1):
                # skip_connection
                decoder = torch.nn.Linear(controller_hid, idx)
                self.decoders.append(decoder)
            # common action
            for idx, size in enumerate(state_space_length):
                decoder = torch.nn.Linear(controller_hid, size)
                self.decoders.append(decoder)
        self._decoders = torch.nn.ModuleList(self.decoders)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, controller_hid),
                cuda,
                requires_grad=False)
        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

        if self.skip_conn:
            self.component_space = component_space_skip
        else:
            self.component_space = component_space

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def append_numClasses(self, actions):
        layers = []
        for action in actions:
            action.append(self.num_classes)
            layers.append(action)
        return layers

    def forward(self,  # pylitorch.nn.Embedding(num_total_tokens,nt:disable=arguments-differ
                inputs,
                hidden,
                block_idx):
        hx, cx = self.lstm(inputs, hidden)
        logits = self.decoders[block_idx](hx)
        logits /= self.softmax_temperature

        # exploration
        if self.mode == 'train':
            logits = (self.tanh_c * F.tanh(logits))

        return logits, (hx, cx)

    def cal_index(self, component, keys, state_length):
        if self.skip_conn:
            start_index = 0
            pop_index = []
            if component == "skip":
                for i in range(1, self.num_layers + 1):
                    skip_index = [start_index + j for j in range(i)]
                    pop_index.extend(skip_index)
                    start_index += state_length + i
                decoder_index = [i for i in range(self.num_layers)]
                embedding_index = [int(i*(state_length+1)) for i in range(self.num_layers)]
            else:
                component_rank = keys.index(component)
                if component == "hidden_units":
                    pop_len = self.num_layers - 1
                else:
                    pop_len = self.num_layers
                for i in range(1, pop_len + 1):
                    component_index = start_index + i + component_rank
                    pop_index.append(component_index)
                    start_index += state_length + i
                decoder_index = component_rank + self.num_layers
                embedding_index = [int(i*(state_length+1)+component_rank+1) for i in range(pop_len)]
        else:
            component_rank = keys.index(component)
            if component == "hidden_units":
                pop_len = self.num_layers - 1
            else:
                pop_len = self.num_layers
            pop_index = [int(i*state_length+component_rank) for i in range(pop_len)]
            decoder_index = component_rank
            embedding_index = pop_index

        return pop_index, decoder_index, embedding_index

    def component_sample(self, batch_size=1, with_details=False, best_actions=None, component='attention'):
        """Samples a set of architectures, via replacing the attention functions in the best_action
        """
        if best_actions is None:
            raise Exception(f'A best_action needs to be given in order to generates better architecture')
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')
        keys = list(state_space.keys())
        state_length = len(state_space)
        keys_skip = keys.append("skip")
        if self.skip_conn and component not in keys_skip:
            raise Exception(f'It is wrong component that does not exist in the search space')
        if not self.skip_conn and component not in keys:
            raise Exception(f'It is wrong component that does not exist in the search space')

        # transform the actions of architecture to indexes of the trainable embedding
        if not isinstance(best_actions[0], list):
            best_actions = [best_actions]
        if self.skip_conn:
            embedding_actions = _deconstruct_action_skip(best_actions, state_space, self.num_tokens, self.num_layers)
        else:
            embedding_actions = _deconstruct_action(best_actions, state_space, self.num_tokens)

        # calculate the indexes of the component that need to be replaced, the decoder index and
        # the embedding index of the component, used to obtain embedding from encoder
        pop_index, decoder_index, embedding_index = self.cal_index(component, keys, state_length)
        # sample the components to replace the original architeture
        multi_entropies = []
        multi_log_probs = []
        multi_actions = []
        for action_i, embedding_action in enumerate(embedding_actions):
            entropies = []
            log_probs = []
            actions = []
            hidden = self.static_init_hidden[batch_size]
            # input variable
            embedding_action = [x for x in embedding_action if x not in pop_index]
            embedding_action = torch.tensor(embedding_action, dtype=torch.long)
            embedding_action = utils.get_variable(embedding_action, self.is_cuda, requires_grad=False)
            embed = self.encoder(embedding_action)
            input = F.normalize(torch.sum(embed, dim=0), p=2, dim=0)
            inputs = torch.stack([input]*batch_size)
            if component == "skip":
                for i in range(self.num_layers):
                    logits, hidden = self.forward(inputs, hidden, decoder_index[i])
                    skip_num = i+1
                    embedding_start = embedding_index[i]
                    entropies, log_probs, actions, inputs = self.skip_sample(logits, entropies,
                                                                             log_probs, actions, embedding_start, skip_num)
            else:
                if component == "hidden_units":
                    pop_len = self.num_layers - 1
                else:
                    pop_len = self.num_layers
                for i in range(pop_len):
                    logits, hidden = self.forward(inputs, hidden, decoder_index)
                    embedding_start = embedding_index[i]
                    entropies, log_probs, actions, inputs = self.action_sample(logits, entropies,
                                                                               log_probs, actions, embedding_start)
            actions = torch.stack(actions).transpose(0, 1)
            log_probs = torch.stack(log_probs).transpose(0, 1)
            entropies = torch.stack(entropies).transpose(0, 1)
            actions = _replace_action(best_actions[action_i], actions, pop_index, component, state_space)
            multi_entropies.append(entropies)
            multi_log_probs.append(log_probs)
            multi_actions.append(actions)

        if with_details:
            return multi_actions, multi_log_probs, multi_entropies
        else:
            return multi_actions

    def sample(self, actions, with_details=False):
        # input is a list of architectures that has different random initiate positions
        # return is a list of architectures that only replace one action in the original architectures
        if not isinstance(actions[0], list):
            actions = [actions]

        multi_actions = []
        multi_log_probs = []
        multi_entropies = []
        components = list(self.component_space.values())
        action_entropies = utils.get_variable(torch.zeros(len(actions), len(self.component_space)),
                                              self.cuda, requires_grad=False)
        for action_i, action in enumerate(actions):
            comSample_actions = []
            comSample_log_probs = []
            comSample_entropies = []
            for com_i, component in enumerate(components):
                comSample_action, comSample_log_prob, comSample_entropy = self.component_sample(
                    batch_size=1,
                    with_details=True, best_actions=action.copy(), component=component)
                comSample_actions.append(comSample_action[0][0])
                comSample_log_probs.append(comSample_log_prob[0][0])
                comSample_entropies.append(comSample_entropy[0][0])
                if self.dataset in ["Cora", "Citeseer", "Pubmed"] and component == 'hidden_units':
                    action_entropies[action_i, com_i] = torch.sum(comSample_entropy[0][0]) * 2.0
                else:
                    action_entropies[action_i, com_i] = torch.sum(comSample_entropy[0][0])
            multi_actions.append(comSample_actions)
            multi_log_probs.append(comSample_log_probs)
            multi_entropies.append(comSample_entropies)
        action_entropies = F.softmax(action_entropies, dim=-1)
        component_sample = action_entropies.multinomial(num_samples=1).data[:,0]
        return_actions = []
        return_log_probs = []
        return_entropies = []
        return_component = []
        for action_i, _ in enumerate(actions):
            return_actions.append(multi_actions[action_i][component_sample[action_i]])
            if with_details:
                return_log_probs.append(multi_log_probs[action_i][component_sample[action_i]])
                return_entropies.append(multi_entropies[action_i][component_sample[action_i]])
                return_component.append(self.component_space[component_sample[action_i].item()])
        if with_details:
            return return_actions, return_log_probs, return_entropies, return_component
        else:
            return return_actions


    def action_sample(self, logits, entropies, log_probs, actions, embedding_start):
        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(log_prob * probs).sum(1, keepdim=False)

        action = probs.multinomial(num_samples=1).data
        selected_log_prob = log_prob.gather(
            1, utils.get_variable(action, requires_grad=False))
        entropies.append(entropy)
        log_probs.append(selected_log_prob[:, 0])
        actions.append(action[:, 0])
        inputs = utils.get_variable(
            action[:, 0] + sum(self.num_tokens[:embedding_start]),
            self.is_cuda,
            requires_grad=False)
        embed = self.encoder(inputs)

        return entropies, log_probs, actions, embed

    def skip_sample(self, logits, entropies, log_probs, actions, embedding_start, skip_num):
        embed = None
        for i in range(skip_num):
            logit_i = logits[:,i]
            logit_i = torch.stack([-logit_i, logit_i], dim=1)
            probs = F.softmax(logit_i, dim=-1)
            log_prob = F.log_softmax(logit_i, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])
            actions.append(action[:, 0])
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:embedding_start]) + 2*i,
                self.is_cuda,
                requires_grad=False)
            if embed is None:
                embed = self.encoder(inputs)
            else:
                embed += self.encoder(inputs)
        return entropies, log_probs, actions, embed

    def init_actions(self, batch_size=1):
        state_length = len(state_space)
        actions = []
        for i in range(len(self.num_tokens)):
            if self.skip_conn:
                skip_flag = True if i%(state_length+1) == 0 else False
            else:
                skip_flag = False
            if skip_flag:
                layer_index = int(self.num_tokens[i]/2)
                for layer_i in range(layer_index):
                    action = [random.randint(0, 1) for _ in range(batch_size)]
                    actions.append(action)
            else:
                action = [random.randint(0, self.num_tokens[i]-1) for _ in range(batch_size)]
                actions.append(action)
        actions = np.transpose(np.stack(actions))
        if self.skip_conn:
            actions = _construct_action_skip(actions, state_space, self.num_layers)
        else:
            actions = _construct_action(actions, state_space)
        actions = self.append_numClasses(actions)
        return actions

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.is_cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.is_cuda, requires_grad=False))


if __name__ == "__main__":
    cntr = MultiController(None, cuda=False)
    print(cntr.init_actions(200))
