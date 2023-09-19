import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils import data
from torch.utils.data import ConcatDataset
import pickle
from sklearn.model_selection import train_test_split
import random
from batch import BatchFinetune, BatchMasking, BatchSubstructContext


# Placeholder implementation of MyDataset
class MyDataset(data.Dataset):
    def __init__(self, node_features, edge_index, edge_attr, subject_features, outcomes):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.subject_features = subject_features
        self.outcomes = outcomes
    
    def __getitem__(self, index):
        node_features_batch = self.node_features[index] 
        edge_index_batch = self.edge_index[index] 
        edge_attr_batch = self.edge_attr[index]
        subject_features_batch = self.subject_features[index] 
        outcomes_batch = self.outcomes[index]
        
        return {
            'node_features': node_features_batch,
            'edge_index': edge_index_batch,
            'edge_attr': edge_attr_batch,
            'subject_features': subject_features_batch,
            'outcomes': outcomes_batch
        }
    
    def __len__(self):
        return len(self.outcomes)
    
    
class MaskDataset(data.Dataset):
    def __init__(self, node_features, edge_index, edge_attr, subject_features, outcomes, masked_edge_idx = None, mask_edge_labels=None):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.subject_features = subject_features
        self.outcomes = outcomes
    
    def __getitem__(self, index):
        node_features_batch = self.node_features[index] 
        edge_index_batch = self.edge_index[index] 
        edge_attr_batch = self.edge_attr[index]
        subject_features_batch = self.subject_features[index] 
        outcomes_batch = self.outcomes[index]
        
        if self.masked_edge_idx is not None and self.mask_edge_labels is not None:
            masked_edge_idx_batch = self.masked_edge_idx[index]
            mask_edge_label_batch = self.mask_edge_labels[index]
            return {'node_features': node_features_batch, 'edge_index': edge_index_batch, 'edge_attr': edge_attr_batch,
            'masked_edge_idx': masked_edge_idx_batch, 'masked_edge_label':mask_edge_label_batch,
            'subject_features': subject_features_batch, 'outcomes': outcomes_batch}
        else:
            return {'node_features': node_features_batch, 'edge_index': edge_index_batch, 'edge_attr': edge_attr_batch,
            'subject_features': subject_features_batch, 'outcomes': outcomes_batch}
        
    def __len__(self):
        return len(self.outcomes)
    
    
def load_data(node_feature_path, edge_feature_path, subject_feature_path, seed):
    
    print("loading finetune data")

    ##first load node features
    with open(node_feature_path, 'rb') as file:
        node_features = pickle.load(file)

    node_list = []
    for node_fea in node_features:
        # Convert each DataFrame to a tensor
        tensor = torch.tensor(node_fea.values, dtype=torch.float32)
        node_list.append(tensor)

    # Stack the tensors to create a higher-dimensional tensor
    node_list_tensor = torch.stack(node_list)


    ##then load edge features
    with open(edge_feature_path, 'rb') as file:
        edge_features = pickle.load(file)

    edge_list = []
    edge_features_list = []
    for df in edge_features:
        i = df["source"].values
        j = df["target"].values
        edge_list.append((i, j))
        
        dim = df.shape[0]
        edge_feature = [df["weight"]]
        edge_feature = np.array(edge_feature, dtype=np.float32).T
        edge_features_list.append(edge_feature)
        
    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float32)
    
    subject_feature = pd.read_csv(subject_feature_path)
    
    subject_fea = subject_feature[["interview_age","traumascore", "efficiency", "assortativity", "transitivity"]]
    tensor_subject_fea = torch.tensor(subject_fea.values, dtype=torch.float32)

    extra_dim = subject_fea.shape[1]
    outcome = subject_feature["alc_sip_ever"]
    tensor_outcome = torch.tensor(outcome.values, dtype=torch.float32)
    
     # Create dataset
    dataset = MyDataset(node_list_tensor, edge_index, edge_attr, tensor_subject_fea, tensor_outcome)
    whole_index = range(len(dataset))
    train_val_index, test_index = train_test_split(whole_index, test_size=0.20, random_state=seed)
    train_index, val_index = train_test_split(train_val_index, test_size=0.25, random_state=seed)
    
    train_subset = data.Subset(dataset, train_index)
    val_subset = data.Subset(dataset, val_index)
    test_subset = data.Subset(dataset, test_index)
    
    subsets = []
    subsets.append((train_subset, val_subset, test_subset))
        
    return subsets, extra_dim


def load_pretain_data(node_feature_path, edge_feature_path, subject_feature_path):
    
    print("loading pratrain data")

    ##first load node features
    with open(node_feature_path, 'rb') as file:
        node_features = pickle.load(file)

    node_list = []
    for node_fea in node_features:
        # Convert each DataFrame to a tensor
        tensor = torch.tensor(node_fea.values, dtype=torch.float32)
        node_list.append(tensor)

    # Stack the tensors to create a higher-dimensional tensor
    node_list_tensor = torch.stack(node_list)

    ##then load edge features
    with open(edge_feature_path, 'rb') as file:
        edge_features = pickle.load(file)

    edge_list = []
    edge_features_list = []
    for df in edge_features:
        i = df["source"].values
        j = df["target"].values
        edge_list.append((i, j))
        
        edge_feature = [df['weight']]
        edge_feature = np.array(edge_feature, dtype=np.float32).T
        edge_features_list.append(edge_feature)
        
    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float32)
    
    subject_feature = pd.read_csv(subject_feature_path)
    
    subject_fea = subject_feature[["interview_age","traumascore", "efficiency", "assortativity", "transitivity"]]
    tensor_subject_fea = torch.tensor(subject_fea.values, dtype=torch.float32)

    extra_dim = subject_fea.shape[1]
    outcome = subject_feature["alc_sip_ever"]
    tensor_outcome = torch.tensor(outcome.values, dtype=torch.float32)
    
     # Create dataset
    dataset = MaskDataset(node_list_tensor, edge_index, edge_attr, tensor_subject_fea, tensor_outcome)
        
    return dataset, extra_dim

def load_data_infomax(node_feature_path, edge_feature_path, subject_feature_path):
    
    print("loading finetune data")

    ##first load node features
    with open(node_feature_path, 'rb') as file:
        node_features = pickle.load(file)

    node_list = []
    for node_fea in node_features:
        # Convert each DataFrame to a tensor
        tensor = torch.tensor(node_fea.values, dtype=torch.float32)
        node_list.append(tensor)

    # Stack the tensors to create a higher-dimensional tensor
    node_list_tensor = torch.stack(node_list)


    ##then load edge features
    with open(edge_feature_path, 'rb') as file:
        edge_features = pickle.load(file)

    edge_list = []
    edge_features_list = []
    for df in edge_features:
        i = df["source"].values
        j = df["target"].values
        edge_list.append((i, j))
        
        dim = df.shape[0]
        edge_feature = [df["weight"]]
        edge_feature = np.array(edge_feature, dtype=np.float32).T
        edge_features_list.append(edge_feature)
        
    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float32)
    
    subject_feature = pd.read_csv(subject_feature_path)
    
    subject_fea = subject_feature[["interview_age","traumascore", "efficiency", "assortativity", "transitivity"]]
    tensor_subject_fea = torch.tensor(subject_fea.values, dtype=torch.float32)

    extra_dim = subject_fea.shape[1]
    outcome = subject_feature["alc_sip_ever"]
    tensor_outcome = torch.tensor(outcome.values, dtype=torch.float32)
    
     # Create dataset
    dataset = MyDataset(node_list_tensor, edge_index, edge_attr, tensor_subject_fea, tensor_outcome)
        
    return dataset, extra_dim

##this func aims to load data for pretrain step2: supervised graph level pretrain
def load_pretrain_data2(node_feature_path, edge_feature_path, subject_feature_path, seed):
    
    print("loading pretrain data for supervised train")

    ##first load node features
    with open(node_feature_path, 'rb') as file:
        node_features = pickle.load(file)

    node_list = []
    for node_fea in node_features:
        # Convert each DataFrame to a tensor
        tensor = torch.tensor(node_fea.values, dtype=torch.float32)
        node_list.append(tensor)

    # Stack the tensors to create a higher-dimensional tensor
    node_list_tensor = torch.stack(node_list)


    ##then load edge features
    with open(edge_feature_path, 'rb') as file:
        edge_features = pickle.load(file)

    edge_list = []
    edge_features_list = []
    for df in edge_features:
        i = df["source"].values
        j = df["target"].values
        edge_list.append((i, j))
        
        edge_feature = [df["weight"]]
        edge_feature = np.array(edge_feature, dtype=np.float32).T
        edge_features_list.append(edge_feature)
        
    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float32)
    
    subject_feature = pd.read_csv(subject_feature_path)
    
    subject_fea = subject_feature[["interview_age","traumascore", "efficiency", "assortativity", "transitivity"]]
    tensor_subject_fea = torch.tensor(subject_fea.values, dtype=torch.float32)

    extra_dim = subject_fea.shape[1]
    outcome = subject_feature["alc_sip_ever"]
    tensor_outcome = torch.tensor(outcome.values, dtype=torch.float32)
    
     # Create dataset
    dataset = MyDataset(node_list_tensor, edge_index, edge_attr, tensor_subject_fea, tensor_outcome)
    whole_index = range(len(dataset))
    train_val_index, test_index = train_test_split(whole_index, test_size=0.20, random_state=seed)
    
    train_val_subset = data.Subset(dataset, train_val_index)
    test_subset = data.Subset(dataset, test_index)
    
    subsets = []
    subsets.append((train_val_subset, test_subset))
        
    return subsets, extra_dim


class MaskEdge:
    def __init__(self, mask_rate):

        self.mask_rate = mask_rate

    def __call__(self, data, masked_edge_indices=None):
        batch_size = data.edge_index.size(0)
        
        if masked_edge_indices == None:
            num_edges = int(data.edge_index.size()[2] / 2)  # num unique edges
            sample_size = int(num_edges * self.mask_rate + 1)
            # during sampling, we only pick the 1st direction of a particular edge pair
            masked_edge_indices = []
            for _ in range(batch_size):
                masked_edge_indices.append([2 * i for i in random.sample(range(num_edges), sample_size)])

        print("calculate masked_edge_idx")
        data.masked_edge_idx = torch.tensor(np.array(masked_edge_indices))

        # create ground truth edge features for the edges that correspond to the masked indices
        mask_edge_labels_list = []
        for i in range(batch_size):
            batch_indices = masked_edge_indices[i]
            edge_attr_i = data.edge_attr[i]
            
            mask_edge_labels_list.append(edge_attr_i[batch_indices, :])
        data.mask_edge_labels = torch.stack(mask_edge_labels_list)
            
        return data


class DataLoaderMasking(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(dataset, batch_size, shuffle, collate_fn=lambda data_list: BatchMasking.from_data_list(data_list), **kwargs)

class DataLoaderFinetune(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderFinetune, self).__init__(dataset, batch_size, shuffle, collate_fn=lambda data_list: BatchFinetune.from_data_list(data_list), **kwargs)

class DataLoaderSubstructContext(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(dataset, batch_size, shuffle, collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list), **kwargs)