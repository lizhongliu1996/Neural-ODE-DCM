import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
import torch.nn as nn

class GINConv2(MessagePassing):
    def __init__(self, emb_dim, aggr="add", input_layer=False):
        super(GINConv2, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_encoder = torch.nn.Linear(1, emb_dim)
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Linear(21, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
        self.aggr = aggr


    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x)
            
        return torch.stack([self.propagate(edge_index = edge_index[i], x = x[i], edge_attr = edge_embeddings[i]) for i in range(x.size(0))])

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim = 1)


    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False
                
            self.gnns.append(GINConv2(emb_dim, aggr = "add", input_layer = input_layer))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, out_dim, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.out_dim)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data):
        x, edge_index, edge_attr = data['node_features'], data['edge_index'], data['edge_attr']
        x, edge_index, edge_attr = torch.stack(x), torch.stack(edge_index), torch.stack(edge_attr)
        node_representation = self.gnn(x, edge_index, edge_attr)
        

        pooled = self.pool(node_representation, batch= None)
        return self.graph_pred_linear(pooled)


##this module will have a ext_x as input, which is the subject phenomic data in our case
class GNN_graphpred2(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, ext_dim, out_dim, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super(GNN_graphpred2, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.ext_dim = ext_dim
        self.out_dim = out_dim
        self.l1_reg = nn.L1Loss(reduction='mean')

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.fc_ext = torch.nn.Linear(self.ext_dim, self.emb_dim)
        self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.out_dim)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data):
        x, edge_index, edge_attr, ext_x = data['node_features'], data['edge_index'], data['edge_attr'], data['subject_features']
        x, edge_index, edge_attr, ext_x = torch.stack(x), torch.stack(edge_index), torch.stack(edge_attr), torch.stack(ext_x)
        node_representation = self.gnn(x, edge_index, edge_attr)

        pooled = self.pool(node_representation, batch= None)
        
        ext_embedding = self.fc_ext(ext_x)
        new_pooled = torch.cat([pooled, ext_embedding], dim = 1)
        output = self.graph_pred_linear(new_pooled)
        l1_loss = self.l1_reg(self.graph_pred_linear.weight, torch.zeros_like(self.graph_pred_linear.weight))

        return output, l1_loss


# class GINConv(MessagePassing):
#     def __init__(self, emb_dim, aggr = "add", input_layer = False):
#         super(GINConv, self).__init__()
#         # multi-layer perceptron
#         self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, 2*emb_dim), 
#                                        torch.nn.BatchNorm1d(2*emb_dim), 
#                                        torch.nn.ReLU(), 
#                                        torch.nn.Linear(2*emb_dim, emb_dim))

#         ### Mapping 0/1 edge features to embedding
#         self.edge_encoder = torch.nn.Linear(3, emb_dim)

#         ### Mapping uniform input features to embedding.
#         self.input_layer = input_layer
#         if self.input_layer:
#             self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
#             torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

#         self.aggr = aggr

#     def forward(self, x, edge_index, edge_attr):
#         #add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes = x.size(1))
        

#         #add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x.size(0), x.size(1), 3)
#         self_loop_attr[:, :,1] = 1 # attribute for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.dtype)
#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

#         edge_embeddings = self.edge_encoder(edge_attr)

#         if self.input_layer:
#             x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

#         return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

#     def message(self, x_j, edge_attr):
#         return torch.cat([x_j, edge_attr], dim = 1)


#     def update(self, aggr_out):
#         return self.mlp(aggr_out)


if __name__ == "__main__":
    pass






