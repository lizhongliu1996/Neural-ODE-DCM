import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils import data
from sklearn.model_selection import KFold, train_test_split

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    if not sp.isspmatrix(mx):
        mx = sp.csr_matrix(mx)

    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    normalized_mx = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    if sp.isspmatrix(normalized_mx):
        return normalized_mx
    else:
        return normalized_mx.A

def read_con_mat(cont_mat_folder):
    adj_ls = []
    n_files = len(os.listdir(cont_mat_folder))
    for i in range(n_files):
        adj = pd.read_csv(cont_mat_folder + f"subject{i}.csv", header = None)
        
        # build symmetric adjacency matrix
        adj = adj + adj.T * (adj.T > adj) - adj * (adj.T > adj)

        # Normalize adj        
        adj = normalize_adj(adj + np.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))  
        
        adj_ls.append(adj)
        
    return torch.stack(adj_ls)

def read_node_fea(node_feature_path):
    node_ls = []
    n_files = len(os.listdir(node_feature_path))
    for i in range(n_files):
        node_fea = pd.read_csv(node_feature_path + f"subject{i}.csv")
        tensor_fea = torch.tensor(node_fea.values, dtype=torch.float32)
        node_ls.append(tensor_fea)
        
    return torch.stack(node_ls)
        
# Placeholder implementation of MyDataset
class MyDataset(data.Dataset):
    def __init__(self, adj_ls, node_feature_ls, subject_feature, outcome):
        self.adj_ls = adj_ls
        self.node_feature_ls = node_feature_ls
        self.subject_feature = subject_feature
        self.outcome = outcome
    
    def __getitem__(self, index):
        adj = self.adj_ls[index]
        node_feature = self.node_feature_ls[index]
        subject_feature = self.subject_feature[index]
        outcome = self.outcome[index]
        return adj, node_feature, subject_feature, outcome
    
    def __len__(self):
        return len(self.outcome)
    
    
def load_data(cont_mat_folder, node_feature_path, subject_feature_path, seed, cv_size = 5):
    
    print("loading data")
    adj_ls = read_con_mat(cont_mat_folder)

    node_feature_ls = read_node_fea(node_feature_path)
    
    #node_feature = torch.FloatTensor(node_info.values.tolist())
    subject_feature = pd.read_csv(subject_feature_path)
    
    subject_fea = subject_feature[["interview_age","traumascore", "efficiency", "assortativity", "transitivity"]]
    tensor_subject_fea = torch.tensor(subject_fea.values, dtype=torch.float32)
    fea_dim = node_feature_ls[0].shape[1]
    extra_dim = subject_fea.shape[1]
    
    outcome = subject_feature["alc_sip_ever"]
    tensor_outcome = torch.tensor(outcome.values, dtype=torch.float64)
    
     # Create dataset
    dataset = MyDataset(adj_ls, node_feature_ls, tensor_subject_fea, tensor_outcome)
    
    # Split data into folds using KFold cross-validation
    kf = KFold(n_splits=cv_size, shuffle=True, random_state=seed)
    fold_index = []
    
    for train_val_index, test_index in kf.split(adj_ls):
        train_index, val_index = train_test_split(train_val_index, test_size=0.25, random_state=seed)
        fold_index.append((train_index, val_index, test_index))
        
    # Create subsets using the fold indices
    subsets = []
    
    #here we are just subset subject level, not node level
    for train_index, val_index, test_index in fold_index:
        train_subset = data.Subset(dataset, train_index)
        val_subset = data.Subset(dataset, val_index)
        test_subset = data.Subset(dataset, test_index)
        
        subsets.append((train_subset, val_subset, test_subset))

    return subsets, fea_dim, extra_dim

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def adj_to_edge_list(adj):
    edge_list = []
    num_nodes = adj.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # To avoid duplicates in an undirected graph
            weight = adj[i][j]
            if abs(weight) >= 0.01:
                edge_list.append((i, j))

    return edge_list


def load_data2(cont_mat_folder, node_feature_path, subject_feature_path, seed, cv_size = 5):
    
    print("loading data")
    adj_ls = read_con_mat(cont_mat_folder)
    edge_ls_all = []
    for adj in adj_ls:
        edge_ls = adj_to_edge_list(adj)
        edge_ls_all.append(edge_ls)
        
    min_len = min(len(lst) for lst in edge_ls_all)
    edge_ls_cut = [lst[:min_len] for lst in edge_ls_all]
    edge_ls_tensor = torch.tensor(edge_ls_cut)
    edge_ls_tensor2 = edge_ls_tensor.permute(0, 2, 1)
    
    node_feature_ls = read_node_fea(node_feature_path)
    
    #node_feature = torch.FloatTensor(node_info.values.tolist())
    subject_feature = pd.read_csv(subject_feature_path)
    
    subject_fea = subject_feature[["interview_age","traumascore", "efficiency", "assortativity", "transitivity"]]
    tensor_subject_fea = torch.tensor(subject_fea.values, dtype=torch.float32)
    fea_dim = node_feature_ls[0].shape[1]
    extra_dim = subject_fea.shape[1]
    
    outcome = subject_feature["alc_sip_ever"]
    tensor_outcome = torch.tensor(outcome.values, dtype=torch.float64)
    
     # Create dataset
    dataset = MyDataset(edge_ls_tensor2, node_feature_ls, tensor_subject_fea, tensor_outcome)
    
    # Split data into folds using KFold cross-validation
    kf = KFold(n_splits=cv_size, shuffle=True, random_state=seed)
    fold_index = []
    
    for train_val_index, test_index in kf.split(adj_ls):
        train_index, val_index = train_test_split(train_val_index, test_size=0.25, random_state=seed)
        fold_index.append((train_index, val_index, test_index))
        
    # Create subsets using the fold indices
    subsets = []
    
    #here we are just subset subject level, not node level
    for train_index, val_index, test_index in fold_index:
        train_subset = data.Subset(dataset, train_index)
        val_subset = data.Subset(dataset, val_index)
        test_subset = data.Subset(dataset, test_index)
        
        subsets.append((train_subset, val_subset, test_subset))

    return subsets, fea_dim, extra_dim

