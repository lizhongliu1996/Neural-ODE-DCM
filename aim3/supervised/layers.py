import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        #adj [32, 77, 77]  e[32, 77, 77] zero_vec[32, 77, 77], wh [32, 77, 128], attention[32,77,77]
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime



    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature) [32, 77, 128]
        # self.a.shape (2 * out_feature, 1) [256, 1]
        # Wh1&2.shape (N, 1) [32, 77, 1]
        # e.shape (N, N)  [77,77]
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
#attention layers for integrations
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nheads):
        super(AttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(input_dim, nheads)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(dim=1)  # Reshape to (batch_size, 1, input_dim)
        attn_output, _ = self.multihead_attention(x, x, x)
        x = attn_output.squeeze(dim=1)  # Remove the extra dimension (batch_size, seq_len, input_dim)
        x = self.fc(x)
        return x
    
    
class GraphAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        #wh [32, 77, 128],
        e = self._prepare_attentional_mechanism_input(Wh, edge_index)
        zero_vec = -9e15 * torch.ones_like(e)
        #edge_index [32, 2, 1574]  e[32, 77, 77] zero_vec[32, 77, 77], attention[32,77,77],  
        attention = torch.where(e > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, edge_index):
        # Wh.shape (batch_size, num_nodes, out_features)
        # self.a.shape (2 * out_features, 1)
        # edge_index.shape (batch_size, 2, num_edges)

        batch_size, num_nodes, out_features = Wh.size()

        # Reshape Wh and apply linear transformations using self.a
        Wh = Wh.view(-1, out_features)  # Reshape Wh to (batch_size * num_nodes, out_features)
        num_edges = edge_index.size(2)

        src_idx = edge_index[:, 0].flatten()  # Shape: (batch_size * num_edges)
        tgt_idx = edge_index[:, 1].flatten()  # Shape: (batch_size * num_edges)

        # Broadcast Wh to get source and target node features for each edge
        src_features = Wh[src_idx]  # Shape: (batch_size * num_edges, out_features)
        tgt_features = Wh[tgt_idx]  # Shape: (batch_size * num_edges, out_features)

        # Concatenate source and target node features along the feature dimension
        a_input = torch.cat([src_features, tgt_features], dim=1)  # Shape: (batch_size * num_edges, 2 * out_features)

        # Compute attention scores using self.a
        e = torch.matmul(a_input, self.a).squeeze()  # Shape: (batch_size * num_edges)

        # Now, we need to create the dense tensor for each graph separately
        e_dense_list = []
        start_idx = 0
        for b in range(batch_size):
            num_edges_b = edge_index[b].size(1)
            end_idx = start_idx + num_edges_b
            e_b = e[start_idx:end_idx]  # Shape: (num_edges_b)
            e_dense_b = torch.zeros(num_nodes, num_nodes, dtype=e.dtype, device=e.device)
            e_dense_b[edge_index[b][0], edge_index[b][1]] = e_b
            e_dense_list.append(e_dense_b)
            start_idx = end_idx

        e_dense = torch.stack(e_dense_list, dim=0)  # Shape: (batch_size, num_nodes, num_nodes)
        return self.leakyrelu(e_dense)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
