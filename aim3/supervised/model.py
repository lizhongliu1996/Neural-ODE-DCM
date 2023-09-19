import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphConvolution, AttentionLayer, GraphAttentionLayer2

    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        x = global_mean_pool(x, batch=None)
        
        return F.log_softmax(x, dim=-1)
    

    

class GAT2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT2, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch=None)
        x = F.elu(self.out_att(x))
        
        return F.log_softmax(x, dim=-1)
    

class GAT3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT3, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = nn.Linear(nhid * nheads, nhid)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch=None)
        x = F.elu(self.out_att(x))
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1)
    
    
    

class GAT_fusion(nn.Module):
    def __init__(self, nfeat, nhid, nfeat_ext, nclass, dropout, alpha, nheads):
        super(GAT_fusion, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_att = nn.Linear(nhid * nheads, nclass)
        self.fc1 = nn.Linear(nfeat_ext, nclass)
        self.fusion = nn.Linear(nclass*2, nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj, sub_fea):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch=None)
        x = F.elu(self.out_att(x))
        
        #append add extra features here
        x_ext = self.fc1(sub_fea)
        x = torch.cat([x, x_ext], dim = 1) 
        x = self.fusion(x)
        
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        
        return F.log_softmax(x, dim=-1), l1_loss
    
    
    
    
   
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        #global_mean_pool is used to extract graph-level information instead of node level
        x = F.relu(global_mean_pool(x, batch=None))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
    
#test for relu
class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        
        x = F.relu(global_mean_pool(x,  batch=None))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
    
#test of selu against relu
class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.selu(self.gc1(x, adj))
        x = F.selu(self.gc2(x, adj))
        
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
    
class GCN4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN4, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  
        self.gc2 = GraphConvolution(nhid, nhid*2)
        self.lc1 = nn.Linear(nhid*2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.selu(self.gc1(x, adj))
        x = F.selu(self.gc2(x, adj))
        
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lc1(x)
        return F.log_softmax(x, dim=1)


#late fusion
class GCN_fusion1(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout):
        super(GCN_fusion1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fusion = nn.Linear(nclass+nfeat_ext, nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj, sub_fea):
        x = F.selu(self.gc1(x, adj))
        x = F.selu(self.gc2(x, adj))
        
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x = torch.cat([x, sub_fea], dim = 1)
        x = self.fusion(x)        
        
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        
        return F.log_softmax(x, dim=1), l1_loss
    
    
#add batch normalization
class GCN_fusion2(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout):
        super(GCN_fusion2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fusion = nn.Linear(nclass+ nfeat_ext, nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')
        self.bn1d = nn.BatchNorm1d(nfeat_ext)

    def forward(self, x, adj, sub_fea):
        x = F.selu(self.gc1(x, adj))
        x = F.selu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x = torch.cat([x, self.bn1d(sub_fea)], dim = 1)
        x = self.fusion(x)        
        
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        
        return F.log_softmax(x, dim=1), l1_loss

#dim of append changes: 2*nhid+nfeat_ext
class GCN_fusion3(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout):
        super(GCN_fusion3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*2)
        self.dropout = dropout
        self.fusion = nn.Linear(nhid*2+nfeat_ext, nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj, sub_fea):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x = torch.cat([x, sub_fea], dim = 1)        
        x = self.fusion(x)
        
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        
        return F.log_softmax(x, dim=1), l1_loss
    
#dim of append changes: nclass+nclass
class GCN_fusion4(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout):
        super(GCN_fusion4, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fc1 = nn.Linear(nfeat_ext, nclass)
        self.fusion = nn.Linear(nclass*2, nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj, sub_fea):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch=None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x_ext = self.fc1(sub_fea)
        x = torch.cat([x, x_ext], dim = 1)        
        x = self.fusion(x)
        
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        
        return F.log_softmax(x, dim=1), l1_loss

#dim of append changes: nhid*2 + nhid
class GCN_fusion5(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout):
        super(GCN_fusion5, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*2)
        self.dropout = dropout
        self.fc1 = nn.Linear(nfeat_ext, nhid)
        self.fusion = nn.Linear(nhid*3, nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj, sub_fea):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x_ext = self.fc1(sub_fea)
        x = torch.cat([x, x_ext], dim = 1)            
        x = self.fusion(x)
        
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        
        return F.log_softmax(x, dim=1), l1_loss


#add instead of append
class GCN_fusion6(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout, ratio):
        super(GCN_fusion6, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc1 = nn.Linear(nfeat_ext, nclass)
        self.dropout = dropout
        self.ratio = ratio

    def forward(self, x, adj, sub_fea):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch= None))
        #direct append add extra features here
        x = F.dropout(x, self.dropout, training=self.training)
        
        x_ext = self.fc1(sub_fea)
        x = x + self.ratio * x_ext
        
        return F.log_softmax(x, dim=1)


##use attention head to fusion
class GCN_fusion7(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout, nheads):
        super(GCN_fusion7, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fc1 = nn.Linear(nfeat_ext, nclass)
        self.attention = AttentionLayer(nclass*2, nclass, nheads)

    def forward(self, x, adj, sub_fea):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x_ext = self.fc1(sub_fea)
        x = torch.cat([x, x_ext], dim = 1)            
        x = self.attention(x)

        return F.log_softmax(x, dim=1)



#change the dimension of gc2 and ext fc to be nhid*2 +nhid
class GCN_fusion8(nn.Module):
    def __init__(self, nfeat, nfeat_ext, nhid, nclass, dropout, nheads):
        super(GCN_fusion8, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*2)
        self.dropout = dropout
        self.fc1 = nn.Linear(nfeat_ext, nhid)
        self.attention = AttentionLayer(nhid*3, nclass, nheads)

    def forward(self, x, adj, sub_fea):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.selu(global_mean_pool(x, batch= None))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #direct append add extra features here
        x_ext = self.fc1(sub_fea)
        x = torch.cat([x, x_ext], dim = 1)            
        x = self.attention(x)

        return F.log_softmax(x, dim=1)
    
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        x = global_mean_pool(x, batch=None)
        
        return F.log_softmax(x, dim=-1)
    
    

class GAT_base(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_base, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer2(nfeat, nhid, dropout=dropout, alpha = alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer2(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        x = global_mean_pool(x, batch=None)
        
        return F.log_softmax(x, dim=1)
    
    
class GCN_base(nn.Module):  
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_base, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        edge_index = edge_index.view(2, -1)
        x = F.relu(self.gc1(x, edge_index))
        x = self.gc2(x, edge_index)
        #global_mean_pool is used to extract graph-level information instead of node level
        x = F.relu(global_mean_pool(x, batch=None))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
    
