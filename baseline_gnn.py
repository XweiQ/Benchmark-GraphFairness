#%%
import dgl
import ipdb
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert

from get_args import *
from load_data import DataLoader, SyntheticGenerator
from models import *
from report import *
from utils import *

def main(args):
    # set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Load data
    data_loader = DataLoader(args)
    dataset = data_loader.load_dataset()    # type <'tuple'>
    adj, features, labels, idx_train, idx_val, idx_test,sens, idx_sens_train = dataset
    if args.model != 'mlp':
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        edge_index = edge_index.to(device)
    
    num_class = labels.unique().shape[0]-1
    if args.dataset in ['nba', 'pokec_z', 'pokec_n']:
        num_class = 1
    compute_statistics(sens, labels, adj, features)
    print('Finish load {}.'.format(args.dataset))

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = False

    # Model and optimizer
    if args.model == 'gcn':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nlayers=args.num_layers,
                    nclass=num_class,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model=="mlp":
        model = MLP(features.shape[1],
                    args.dropout,
                    args.hidden,
                    num_class,
                    args.num_layers)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    # Train model
    features = features.to(device)
    labels = labels.to(device)
    acc_val_list = []
    roc_val_list = []
    f1_val_list = []
    parity_val_list = []
    equality_val_list = []
    acc_test_list = []
    fair_list = []
    roc_test_list = []
    f1_test_list = []
    parity_list = []
    equality_list = []
    groups_acc_list = []
    loss_list = []

    for epoch in range(args.epochs+1):

        if args.model in ['mlp']:
            model.train()
            optimizer.zero_grad()
            output = model(features)
            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(labels)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately,
            model.eval()
            output = model(features)
            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
            val_loss = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val])
            auc_roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
            f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())

            acc_val_list.append(acc_val.item())
            roc_val_list.append(auc_roc_val.item())
            f1_val_list.append(f1_val.item())
            parity_val_list.append(parity_val)
            equality_val_list.append(equality_val)
            loss_list.append(val_loss.item())

            # test
            output_preds = (output.squeeze()>0).type_as(labels)
            acc_test = accuracy(output_preds[idx_test], labels[idx_test])
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            groups_acc = group_acc(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            
            acc_test_list.append(acc_test.item())
            roc_test_list.append(auc_roc_test.item())
            f1_test_list.append(f1_s.item())
            parity_list.append(parity)
            equality_list.append(equality)
            groups_acc_list.append(groups_acc)

        elif args.model in ['gcn', 'sage', 'gat']:
            model.train()
            optimizer.zero_grad()
            output = model(features, edge_index)
            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(labels)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately,
            model.eval()
            output = model(features, edge_index)
            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
            val_loss = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val])
            auc_roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
            f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())

            acc_val_list.append(acc_val.item())
            roc_val_list.append(auc_roc_val.item())
            f1_val_list.append(f1_val.item())
            parity_val_list.append(parity_val)
            equality_val_list.append(equality_val)
            fair_list.append(parity_val+equality_val)
            loss_list.append(val_loss.item())

            # test
            output_preds = (output.squeeze()>0).type_as(labels)
            acc_test = accuracy(output_preds[idx_test], labels[idx_test])
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            groups_acc = group_acc(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())

            acc_test_list.append(acc_test.item())
            roc_test_list.append(auc_roc_test.item())
            f1_test_list.append(f1_s.item())
            parity_list.append(parity)
            equality_list.append(equality)
            groups_acc_list.append(groups_acc)

    history = {
        'args': args,
        'val_acc': acc_val_list,
        'test_acc': acc_test_list,
        'val_roc': roc_val_list,
        'test_roc': roc_test_list,
        'val_f1': f1_val_list,
        'test_f1': f1_test_list,
        'val_parity': parity_val_list,
        'val_equality': equality_val_list,
        'test_parity': parity_list,
        'test_equality': equality_list,
        'val_loss': loss_list,
        'group_acc_list': groups_acc_list, 
    }

    return history, groups_acc_list

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    args = get_baseline_args(parser)
    # set the result path
    base_path = os.path.join('./'+args.dataset, args.model, 'layers'+str(args.num_layers), 
                        'lr'+str(args.lr)+'wd'+str(args.weight_decay), 'drop'+str(args.dropout))
    create_directory_safely(base_path)

    t_total = time.time()

    if args.task == 'train':
        if os.path.exists(base_path+'/seed'+str(args.seed)+'history.pkl'): 
            print("this result is existing") 
            exit()
        else:
            history, groups_acc_list = main(args)
            with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'wb') as file:
                pickle.dump(history, file)
            evaluate(args, base_path, groups_acc_list)
    elif args.task == 'eva':
        toExcel(args, base_path)
        report(args, base_path, excel=True)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
