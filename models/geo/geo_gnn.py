import torch
import torch.nn.functional as F

import models.geo.utils as util
from models.geo.geo_layer import GeoLayer
from models.gnn import GraphNet as BaseNet
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot


class GraphNet(BaseNet):
    '''
    not contain jump knowledge layer
    '''

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=False, state_num=5,
                 residual=False, layer_nums=3, dataset='Cora'):
        self.residual = residual
        self.batch_normal = batch_normal
        self.dataset = dataset
        super(GraphNet, self).__init__(actions, num_feat, num_label, drop_out, multi_label, batch_normal, residual,
                                       state_num, layer_nums)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.skip_connections = []
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)
        # if self.dataset in ["Cora", "Citeseer", "Pubmed"]:
        #     self.input_transform = Parameter(torch.Tensor(self.num_feat, 128))
        #     glorot(self.input_transform)

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=5):

        # build hidden layer
        for i in range(layer_nums):
            if i == 0:
                in_channels = num_feat
                # in_channels = 128
            else:
                in_channels = out_channels * head_num

            # extract layer information
            if not self.residual:
                attention_type = actions[i * state_num + 0]
                aggregator_type = actions[i * state_num + 1]
                act = actions[i * state_num + 2]
                head_num = actions[i * state_num + 3]
                update_type = actions[i * state_num + 4]
                out_channels = actions[i * state_num + 5]
            else:
                start_index = i * state_num + int((i+1) * i / 2)
                skip_index = actions[start_index:start_index+i+1] if i != 0 else [actions[start_index]]
                self.skip_connections.append(skip_index)
                attention_type = actions[start_index+i+1]
                aggregator_type = actions[start_index+i+2]
                act = actions[start_index+i+3]
                head_num = actions[start_index+i+4]
                update_type = actions[start_index+i+5]
                out_channels = actions[start_index+i+6]

            concat = True
            if i == layer_nums - 1:
                concat = False
            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                                        att_type=attention_type, agg_type=aggregator_type, update_type=update_type,
                                        act_type=act, dataset=self.dataset))
            self.acts.append(util.act_map(act))
            if concat:
                self.fcs.append(torch.nn.Linear(in_channels, out_channels * head_num))
            else:
                self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index_all):
        if self.dataset == 'PPI':
            output = x
        else:
            # output = torch.mm(x, self.input_transform)
            # output = F.normalize(output, p=2, dim=-1)
            output = x

        if self.residual:
            layer_outputs = [output]
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = layer(output, edge_index_all)
                for j in range(len(layer_outputs)):
                    layer_outputs[j] = fc(layer_outputs[j])
                    if self.skip_connections[i][j] == 1:
                        output += layer_outputs[j]
                    else:
                        pass
                output = act(output)
                layer_outputs.append(output)
        else:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                if self.dataset in ['Cora', 'Citeseer', 'Pubmed']:
                    output = layer(output, edge_index_all)
                else:
                    output = layer(output, edge_index_all) + fc(output) # if args.dataset='PPI'
                output = act(output)
        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param

        return result

    def load_param(self, param):
        if param is None:
            return

        for i in range(self.layer_nums):
            key = "layer_%d" % i
            if key in param:
                self.layers[i].load_param(param[key])

