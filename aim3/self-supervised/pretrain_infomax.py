import argparse

from utils import load_data_infomax

from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

import pandas as pd


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class Infomax(nn.Module):
    def __init__(self, gnn, discriminator):
        super(Infomax, self).__init__()
        self.gnn = gnn
        self.discriminator = discriminator
        self.loss = nn.BCEWithLogitsLoss()
        self.pool = global_mean_pool


def train(args, model, loader, optimizer):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, edge_index, edge_attr = batch['node_features'], batch['edge_index'], batch['edge_attr']
        node_emb = model.gnn(x, edge_index, edge_attr)
        summary_emb = torch.sigmoid(node_emb)

        positive_expanded_summary_emb = summary_emb

        shifted_summary_emb = summary_emb[cycle_index(len(summary_emb), 1)]
        negative_expanded_summary_emb = shifted_summary_emb

        positive_score = model.discriminator(node_emb, positive_expanded_summary_emb)
        negative_score = model.discriminator(node_emb, negative_expanded_summary_emb)      

        optimizer.zero_grad()
        loss = model.loss(positive_score, torch.ones_like(positive_score)) + model.loss(negative_score, torch.zeros_like(negative_score))
        loss.backward()

        optimizer.step()

        train_loss_accum += float(loss.detach().item())
        acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        train_acc_accum += float(acc.detach().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)


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
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type = str, default = 'pretrain_model/pretrain_infomax', help='filename to output the pre-trained model')
    parser.add_argument('--seed', type=int, default=123, help = "Seed for splitting dataset.")
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #set up dataset
    node_feature_path = "pretrain_data/node_fea.pkl"
    edge_feature_path = "pretrain_data/edge_fea.pkl"
    subject_feature_path = "pretrain_data/sub_fea.csv"
    dataset, extra_dim = load_data_infomax(node_feature_path, edge_feature_path, subject_feature_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio)

    discriminator = Discriminator(args.emb_dim)

    model = Infomax(gnn, discriminator)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, loader, optimizer)

        print(train_acc)
        print(train_loss)


    if not args.model_file == "":
        torch.save(model.gnn.state_dict(), args.model_file + ".pth")

if __name__ == "__main__":
    main()
