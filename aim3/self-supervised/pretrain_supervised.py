import argparse

from utils import load_pretrain_data2, DataLoaderFinetune
# from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

import pandas as pd


criterion = nn.BCEWithLogitsLoss()

def train(args, model, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        pred = model(batch)
        y = torch.tensor(batch['outcomes']).view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()

        loss_accum += loss.detach()

    return loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2, help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--input_model_file', type=str, default = 'pretrain_model/pretrain_infomax', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = 'pretrain_model/infomax_supervised', help='filename to output the pre-trained model')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    node_feature_path = "pretrain_data/node_fea.pkl"
    edge_feature_path = "pretrain_data/edge_fea.pkl"
    subject_feature_path = "pretrain_data/sub_fea.csv"
    data, extra_dim = load_pretrain_data2(node_feature_path, edge_feature_path, subject_feature_path, args.seed)

    training_val_set, test_set = data[0][0], data[0][1]
    train_loader = DataLoaderFinetune(training_val_set, batch_size=args.batch_size, shuffle=True)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, out_dim = 1, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling)
    if not args.input_model_file == "":
        print("loading pretrain model")
        model.from_pretrained(args.input_model_file + ".pth")
    

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)   

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_loss = train(args, model, train_loader, optimizer)
        print("train loss:", train_loss)

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")



if __name__ == "__main__":
    main()
