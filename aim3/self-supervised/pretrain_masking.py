import argparse
from utils import load_pretain_data, DataLoaderMasking

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
from model import GNN
from utils import MaskEdge

criterion = nn.MSELoss()

def compute_rmse(pred, target):
    squared_diff = torch.pow(pred - target, 2)
    mse = torch.mean(squared_diff)
    rmse = torch.sqrt(mse)
    return rmse.item()

def train(args, model_list, loader, optimizer_list):
    model, linear_pred_edges = model_list
    optimizer_model, optimizer_linear_pred_edges = optimizer_list

    model.train()
    linear_pred_edges.train()

    loss_accum = 0
    rmse_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        
        x, edge_index, edge_attr = torch.stack(batch['node_features']), torch.stack(batch['edge_index']), torch.stack(batch['edge_attr'])
        print(edge_index.shape)
        print(edge_index[0].shape)
        masked_edge_idx, mask_edge_label = torch.stack(batch['masked_edge_idx']), torch.stack(batch['masked_edge_label'])
        node_rep = model(x, edge_index, edge_attr)

        ### predict the edge types.
        masked_edge_index = []
        for i in range(edge_index.shape[0]):
            edge_index_i, masked_edge_idx_i = edge_index[i], masked_edge_idx[i]
            masked_edge_index.append(edge_index_i[:, masked_edge_idx_i])
            
        edge_rep = []
        for i in range(len(masked_edge_index)):
            node_rep_i, masked_edge_index_i = node_rep[i], masked_edge_index[i]
            edge_rep.append(node_rep_i[masked_edge_index_i[0]] + node_rep_i[masked_edge_index_i[1]])

        pred_edge = linear_pred_edges(torch.stack(edge_rep))
       
        #converting the binary classification to multiclass classification
        rmse_edge = compute_rmse(pred_edge, mask_edge_label)
        rmse_accum += rmse_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_edges.zero_grad()
        
        loss = criterion(pred_edge, mask_edge_label)
        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_edges.step()

        loss_accum += float(loss.item())

    return loss_accum/(step + 1), rmse_accum/(step + 1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15, help='dropout ratio (default: 0.15)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default = 'pretrain_model/pretrain_mask', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=1, help = "Seed for splitting dataset.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("num layer: %d mask rate: %f" %(args.num_layer, args.mask_rate))
    
    #set up dataset
    node_feature_path = "pretrain_data/node_fea.pkl"
    edge_feature_path = "pretrain_data/edge_fea.pkl"
    subject_feature_path = "pretrain_data/sub_fea.csv"
    dataset, extra_dim = load_pretain_data(node_feature_path, edge_feature_path, subject_feature_path)
    ##apply transforms on data
    mask_edge = MaskEdge(mask_rate = args.mask_rate)
    modified_data  = mask_edge(dataset)

    loader = DataLoaderMasking(modified_data, batch_size=args.batch_size, shuffle=True)

    #set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio)
    #Linear layer for classifying different edge types
    linear_pred_edges = torch.nn.Linear(args.emb_dim, 1)
    model_list = [model, linear_pred_edges]

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_edges = optim.Adam(linear_pred_edges.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_edges]

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_rmse = train(args, model_list, loader, optimizer_list)
        print("train loss:", train_loss, "train rmse:", train_rmse)

    if not args.model_file == "":
        torch.save(model.state_dict(), args.model_file + ".pth")


if __name__ == "__main__":
    main()
