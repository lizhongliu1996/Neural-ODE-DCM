import argparse

from utils import load_data, DataLoaderFinetune
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
import pandas as pd

import os
import pickle

criterion = nn.BCEWithLogitsLoss()

def train(args, model, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        pred = model(batch)
        y = torch.stack(batch['outcomes']).view(pred.shape).to(torch.float64)
        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()

        return loss.item()


def eval(args, model, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        with torch.no_grad():
            pred = model(batch)

        y_true.append(torch.stack(batch['outcomes']).view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
        else:
            roc_list.append(np.nan)

    return np.array(roc_list) #y_true.shape[1]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default = 'pretrain_model/infomax_supervised', help='filename to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=123, help = "Seed for running experiments.")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    node_feature_path = "finetune_data/node_fea.pkl"
    edge_feature_path = "finetune_data/edge_fea.pkl"
    subject_feature_path = "finetune_data/sub_fea.csv"

    data, extra_dim = load_data(node_feature_path, edge_feature_path, subject_feature_path, seed = args.seed)

    training_set, validation_set, test_set = data[0][0], data[0][1], data[0][2]

    train_loader = DataLoaderFinetune(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoaderFinetune(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoaderFinetune(test_set, batch_size=args.batch_size, shuffle=False) 

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, out_dim = 1, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling)

    if not args.model_file == "":
        model.from_pretrained(args.model_file + ".pth")

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    loss_list = []
    

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss = train(args, model, train_loader, optimizer)
        loss_list.append(train_loss)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, train_loader)
        else:
            train_acc = 0
            print("ommitting training evaluation")
        val_acc = eval(args, model, val_loader)

        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        
        test_acc = eval(args, model, test_loader)
        test_acc_list.append(test_acc)

        print("")

    def find_max_values(list_of_lists):
        max_values = []

        for sublist in list_of_lists:
            sublist_max = np.max(sublist)
            max_values.append(sublist_max)

        return max_values
    
    max_test = max(find_max_values(test_acc))
    print("best test roc is: ", max_test)

    def draw_loss_graph(loss_values, save_path):
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'b-', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()

        plt.savefig(save_path)
        plt.close()

    # Example usage:
    save_path = 'loss_graph_full.png'

    draw_loss_graph(loss_list, save_path)
    
    os.makedirs(f"result/finetune_seed{args.seed}" , exist_ok=True)
    filepath = f"result/finetune_seed{args.seed}"+"/"+"fullmodel_info_result.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump({"train": np.array(train_acc_list), "val": np.array(val_acc_list), "test": np.array(test_acc_list)}, f)
            

if __name__ == "__main__":
    main()