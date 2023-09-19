import torch

import numpy as np
import random

import argparse
import time
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from model import GCN_fusion7
from utils import load_data, accuracy
import glob
import json
import os
import time
    
parser = argparse.ArgumentParser()


# # hyper-parameters
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--nheads', type=int, default=1, help='nums for multihead attention')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--cv_size', type=int, default=5, help='size of inner cv')

args = parser.parse_args()
is_cuda = False

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

######1. load data
cont_mat_path = "../data2/cont_mats/"
node_feature_path = "../data2/node_features2/"
subject_feature_path = "../data2/sub_features.csv"
data_cv, num_features, num_features_extra = load_data(cont_mat_path, node_feature_path, subject_feature_path, seed, args.cv_size)

sv_path = "../results/gcn_fusion7/"
patience = 50

for i in range(len(data_cv)):
    print(f"CV_{i}")
    training_set, validation_set, test_set = data_cv[i][0], data_cv[i][1], data_cv[i][2]
    
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=len(validation_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False) 

        
    ######2. create model
    model = GCN_fusion7(nfeat = num_features,
                nfeat_ext= 5,
                nhid = args.nhid,
                nclass=2,
                dropout=args.dropout,
                nheads = args.nheads
                )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            
    if is_cuda:
        model.cuda()
        train_loader.cuda()
        val_loader.cuda()
        test_loader.cuda()
        
    ######3. define train function and test function
    
    ##to shuffle data, just need to reorder the index of subject_feature and order of adj matrix with node features
    #time wasting, doing later
    def compute_valid(loader):
        model.eval()
        loss_val = 0.0
        output_total, outcome_total = [], []
        with torch.no_grad():
            for data in loader:
                adj, node_feature, subject_feature, outcome = data
                out = model(node_feature, adj, subject_feature)
                loss_val += F.nll_loss(out, outcome.long()).item()
                outcome_total.extend(outcome)
                output_total.extend(out)

        acc_val = accuracy(torch.stack(output_total), torch.stack(outcome_total))
        loss_val = loss_val/len(loader)    
                
        return loss_val, acc_val
    
    def train(epoch, train_loader, val_loader):
        t = time.time()
        loss_train = 0.0
        output_total, outcome_total = [], []
        for i, data in enumerate(train_loader):
            adj, node_feature, subject_feature, outcome = data
            model.train()
            optimizer.zero_grad()
            output = model(node_feature, adj, subject_feature)
            loss = F.nll_loss(output, outcome.long())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            outcome_total.extend(outcome)
            output_total.extend(output)
        
        acc_train = accuracy(torch.stack(output_total), torch.stack(outcome_total))
        loss_train = loss_train/len(train_loader)
        loss_val, acc_val = compute_valid(val_loader)
        
        print('Epoch: {:03d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train),
            'acc_train: {:.4f}'.format(acc_train),
            'loss_val: {:.4f}'.format(loss_val),
            'acc_val: {:.4f}'.format(acc_val),
            'time: {:.4f}s'.format(time.time() - t))

        return loss_val



    def compute_test(model, loader):
        model.eval()
        loss_test = 0.0
        output_total, outcome_total, out_prob = [], [], []
        with torch.no_grad():
            for data in loader:
                adj, node_feature, subject_feature, outcome = data
                out = model(node_feature, adj, subject_feature)
                y_prob = torch.exp(out)[:, 1].detach().numpy()
                loss_test += F.nll_loss(out, outcome.long()).item()
                outcome_total.extend(outcome)
                output_total.extend(out)
                out_prob.extend(y_prob)

        acc_test = accuracy(torch.stack(output_total), torch.stack(outcome_total))
        loss_test = loss_test/len(loader)
        
        outcome_ls = [tensor.item() for tensor in outcome_total]
        auc_score = roc_auc_score(outcome_ls, out_prob)
        y_pred = np.array([0 if p < 0.5 else 1 for p in out_prob])
        false_alarm_rate = np.mean(y_pred == 1)
        f1score = f1_score(outcome_total, y_pred)
        
        print("Test set results:",
            "loss= {:.4f}".format(loss_test),
            "accuracy= {:.4f}".format(acc_test),
            "auc= {:.4f}".format(auc_score),
            "False Alarm Rate = {:.4f}".format(false_alarm_rate),
            "F1 Score = {:.4f}".format(f1score))
    
        return loss_test, round(acc_test.item(), 4), round(auc_score, 4), round(false_alarm_rate, 4), f1score


######4. train the model
    t_total = time.time()
    bad_counter = 0
    best_epoch = 0
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        loss = train(epoch, train_loader, val_loader)

        torch.save(model.state_dict(), sv_path + '{}.pkl'.format(epoch))
        #need to work on this 
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    files = glob.glob(sv_path +'*.pkl')
    for file in files:
        filename = file.split('/')[3]
        #need to modify this on linux
        #fname = filename.split('\\')[1]
        epoch_nb = int(filename.split('.')[0])
        if epoch_nb != best_epoch:
            os.remove(file)
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(sv_path +'{}.pkl'.format(best_epoch)))

    test_loss, acc_test, auc, far, f1 = compute_test(model, test_loader)
    
    hyper_para = {}
    hyper_para["cv"] = i
    hyper_para["batch"] = args.batch_size
    hyper_para["lr"] = args.lr
    hyper_para["weight_decay"] = args.weight_decay
    hyper_para["dropout"] = args.dropout
    hyper_para["hidden_dim"] = args.nhid
    hyper_para["nheads"] = args.nheads
    hyper_para["loss"] = test_loss
    hyper_para["accuracy"] = acc_test
    hyper_para["auc"] = auc
    hyper_para["false_alarm_rate"] = far
    hyper_para["f1_score"] = f1
    with open(sv_path + "hyperpara.json", "a+") as fp:
        fp.write('\n')
        json.dump(hyper_para, fp) 
