"""
encoder for ImGDA
"""
import torch
from torch import nn
from GNN.ppmi_conv import PPMIConv
import torch.nn.functional as F

class add_compensation(nn.Module):
    def __init__(self, nodes_num, encoder_dim, device):
        super(add_compensation, self).__init__()
        self.compensation = nn.Parameter(torch.FloatTensor(nodes_num, encoder_dim).normal_(-1,1).to(device))
        self.compensation.requires_grad_(True)

    def forward(self, input):
        return input + self.compensation

class GNNEncoder(nn.Module):
    def __init__(self, source_data, encoder_dim, drop_out, feature_num, device, **kwargs):
        super(GNNEncoder, self).__init__()
        weights = [None, None]
        biases = [None, None]

        self.dropout_layers = [nn.Dropout(drop_out) for _ in weights]

        self.compensation_layers = nn.ModuleList(
            [add_compensation(source_data[0], encoder_dim, device),
             add_compensation(source_data[0], encoder_dim, device),]
        )

        self.conv_layers = nn.ModuleList([
            PPMIConv(feature_num, encoder_dim, weight=weights[0], bias = biases[0], **kwargs),
            PPMIConv(feature_num, encoder_dim, weight=weights[1], bias = biases[1], **kwargs),
        ]).to(device)

    def forward(self, x, edge_index, cache_name, compensate, device, use_ppmi = True):
        x = x.to(device)
        edge_index = edge_index.to(device)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, use_ppmi = use_ppmi, cache_name= cache_name)
            if compensate:
                x = self.compensation_layers[i](x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x
