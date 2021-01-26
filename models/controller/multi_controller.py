"""A module with NAS controller-related code."""

import torch
import torch.nn.functional as F
import numpy as np
import utils
from torch.autograd import Variable
import random


state_space = {
    "attention_type": ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear'],
    'aggregator_type': ["add", "mean", "max"],  # remove lstm
    'activate_function': ["sigmoid", "tanh", "relu", "linear",
                          "softplus", "leaky_relu", "relu6", "elu"],
    'number_of_heads': [1, 2, 4, 6, 8, 16],
    'update_type': ['identity', 'mlp'],
    'hidden_units': [4, 8, 16, 32, 64, 128, 256],
}


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

def _replace_action_sequential(best_action, actions, pop_indexs, components, state_space):
    # given the best_action, using the action in actions to replace the original architecture sequentially
    predicted_action = best_action
    for action_i, action in enumerate(actions):
        pop_index = pop_indexs[action_i]
        component = components[action_i]
        for i in range(len(pop_index)):
            if component == "skip":
                predicted_action[pop_index[i]] = action[i]
            else:
                predicted_action[pop_index[i]] = state_space[component][action[i]]
    return predicted_action


class MultiController(torch.nn.Module):

    def __init__(self, args, num_layers=3, skip_conn=False, controller_hid=100, cuda=True, mode="train",
                 softmax_temperature=5.0, tanh_c=2.5, num_classes=121, num_selectCom = 1):
        torch.nn.Module.__init__(self)
        self.mode = mode
        self.num_layers = num_layers
        self.skip_conn = skip_conn
        self.controller_hid = controller_hid
        self.num_classes = num_classes
        self.is_cuda = cuda
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
        self.num_selectCom = num_selectCom

        # compute relative index
        self.pop_indexs, self.decoder_indexs, self.embedding_indexs = self.init_index()

    def init_index(self):
        pop_indexs = {}
        decoder_indexs = {}
        embedding_indexs = {}
        components = list(self.component_space.values())
        keys = list(state_space.keys())
        state_length = len(state_space)
        for component in components:
            pop_index, decoder_index, embedding_index = self.cal_index(component, keys, state_length)
            pop_indexs[component] = pop_index
            decoder_indexs[component] = decoder_index
            embedding_indexs[component] = embedding_index
        return pop_indexs, decoder_indexs, embedding_indexs

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
                embedding_index = [int(i * (state_length + 1)) for i in range(self.num_layers)]
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
                embedding_index = [int(i * (state_length + 1) + component_rank + 1) for i in range(pop_len)]
        else:
            component_rank = keys.index(component)
            if component == "hidden_units":
                pop_len = self.num_layers - 1
            else:
                pop_len = self.num_layers
            pop_index = [int(i * state_length + component_rank) for i in range(pop_len)]
            decoder_index = component_rank
            embedding_index = pop_index

        return pop_index, decoder_index, embedding_index

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

    def component_sample(self, batch_size=1, with_details=False, best_actions=None, component='attention'):
        """Samples a set of architectures, via replacing the attention functions in the best_action
        """
        if best_actions is None:
            raise Exception(f'A best_action needs to be given in order to generates better architecture')
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')
        if component not in list(self.component_space.values()):
            raise Exception(f'It is wrong component that does not exist in the search space')

        # transform the actions of architecture to indexes of the trainable embedding
        if not isinstance(best_actions[0], list):
            best_actions = [best_actions]
        if self.skip_conn:
            embedding_actions = _deconstruct_action_skip(best_actions, state_space, self.num_tokens, self.num_layers)
        else:
            embedding_actions = _deconstruct_action(best_actions, state_space, self.num_tokens)

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
            embedding_action = [x for x in embedding_action if x not in self.pop_indexs[component]]
            embedding_action = torch.tensor(embedding_action, dtype=torch.long)
            embedding_action = utils.get_variable(embedding_action, self.is_cuda, requires_grad=False)
            embed = self.encoder(embedding_action)
            input = F.normalize(torch.sum(embed, dim=0), p=2, dim=0)
            inputs = torch.stack([input]*batch_size)

            if component == "skip":
                for i in range(self.num_layers):
                    logits, hidden = self.forward(inputs, hidden, self.decoder_indexs[component][i])
                    skip_num = i+1
                    embedding_start = self.embedding_indexs[component][i]
                    entropies, log_probs, actions, inputs = self.skip_sample(logits, entropies,
                                                                             log_probs, actions, embedding_start, skip_num)
            else:
                if component == "hidden_units":
                    pop_len = self.num_layers - 1
                else:
                    pop_len = self.num_layers
                for i in range(pop_len):
                    logits, hidden = self.forward(inputs, hidden, self.decoder_indexs[component])
                    embedding_start = self.embedding_indexs[component][i]
                    entropies, log_probs, actions, inputs = self.action_sample(logits, entropies,
                                                                               log_probs, actions, embedding_start)
            actions = torch.stack(actions).transpose(0, 1)
            log_probs = torch.stack(log_probs).transpose(0, 1)
            entropies = torch.stack(entropies).transpose(0, 1)
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

        component_actions = []
        component_log_probs = []
        component_entropies = []
        components = list(self.component_space.values())
        action_entropies = utils.get_variable(torch.zeros(len(actions), len(self.component_space)),
                                              self.cuda, requires_grad=False)

        # compute the action, prob and entropy
        for com_i, component in enumerate(components):
            multi_actions, multi_log_probs, multi_entropies = self.component_sample(batch_size=1,with_details=True,
                                                                                    best_actions=actions,
                                                                                    component=component)
            component_actions.append(multi_actions)
            component_log_probs.append(multi_log_probs)
            component_entropies.append(multi_entropies)
            sum_extropy = torch.sum(torch.stack(multi_entropies), dim=-1).squeeze(1)
            action_entropies[:, com_i] = sum_extropy

        # sample the action
        action_entropies = F.softmax(action_entropies, dim=-1)
        selected_coms = []
        for i in range(self.num_selectCom):
            component_sample = action_entropies.multinomial(num_samples=1).data[:, 0]
            action_entropies[list(range(len(actions))), component_sample] = 0
            selected_coms.append(component_sample)
        selected_coms = torch.stack(selected_coms).transpose(0, 1)

        return_actions = []
        return_log_probs = []
        return_entropies = []
        return_components = []
        for action_i, action in enumerate(actions):
            selected_actions = []
            selected_log_probs = []
            selected_entropies = []
            selected_components = []
            pop_indexs = []
            for i in range(self.num_selectCom):
                component = selected_coms[action_i, i].item()
                selected_components.append(self.component_space[component])
                selected_entropies.append(component_entropies[component][action_i][0])
                selected_log_probs.append(component_log_probs[component][action_i][0])
                selected_actions.append(component_actions[component][action_i][0])
                pop_indexs.append(self.pop_indexs[selected_components[-1]])
            return_components.append(selected_components)
            return_entropies.append(selected_entropies)
            return_log_probs.append(selected_log_probs)
            recons_action = _replace_action_sequential(action.copy(), selected_actions,
                                                       pop_indexs, selected_components, state_space)
            return_actions.append(recons_action)

        if with_details:
            return return_actions, return_log_probs, return_entropies, return_components
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
