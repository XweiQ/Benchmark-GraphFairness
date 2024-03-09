import dgl
import time
import tqdm
import ipdb
import pickle
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_adj, convert
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score

from get_args import *
from load_data import DataLoader, SyntheticGenerator
from models import *
from report import *
from utils import *

def train(model, x, edge_index, labels, idx_train, sens, idx_sens_train):
    model.train()
    G_params = list(model.GNN.parameters()) + list(model.classifier.parameters()) + list(model.estimator.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
    optimizer_A = torch.optim.Adam(model.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_g_loss = 0
    train_a_loss = 0

    ### update E, G
    model.adv.requires_grad_(False)
    optimizer_G.zero_grad()

    s = model.estimator(x, edge_index)
    h = model.GNN(x, edge_index)
    y = model.classifier(h)

    s_g = model.adv(h)
    s_score_sigmoid = torch.sigmoid(s.detach())
    s_score = s.detach()
    s_score[idx_train]=sens[idx_train].unsqueeze(1).float()
    y_score = torch.sigmoid(y)
    cov =  torch.abs(torch.mean((s_score_sigmoid[idx_train] - torch.mean(s_score_sigmoid[idx_train])) * (y_score[idx_train] - torch.mean(y_score[idx_train]))))
    
    cls_loss = criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
    adv_loss = criterion(s_g[idx_train], s_score[idx_train])
    G_loss = cls_loss  + args.alpha * cov - args.beta * adv_loss
    G_loss.backward()
    optimizer_G.step()

    ## update Adv
    model.adv.requires_grad_(True)
    optimizer_A.zero_grad()
    s_g = model.adv(h.detach())
    A_loss = criterion(s_g[idx_train], s_score[idx_train])
    A_loss.backward()
    optimizer_A.step()
    return G_loss.detach().cpu + A_loss.detach().cpu()

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
    G = dgl.DGLGraph()    # dgl 0.4.3
    G.from_scipy_sparse_matrix(adj)
    # G = dgl.from_scipy(adj)
    # G = G.to(torch.device('cuda:0'))
    compute_statistics(sens, labels, adj, features)
    print('Finish load {}.'.format(args.dataset))

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = False

    # Model and optimizer
    model = FairGNN(nfeat = features.shape[1], args = args).to(device)
    if args.cuda:
        model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)
        sens = sens.to(device)
        idx_sens_train = idx_sens_train.to(device)

    # Train model
    features = features.to(device)
    edge_index = edge_index.to(device)
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
    loss_list = []
    groups_acc_list = []

    for epoch in range(args.epochs+1):
        model.train()
        model.optimize(G,features,labels,idx_train,sens,idx_sens_train)
        loss = model.G_loss + model.A_loss
        
        model.eval()
        output, ss, z = model(features, G)
        output_preds = (output.squeeze()>0).type_as(labels)
        # validation
        acc_val = accuracy(output_preds[idx_val], labels[idx_val]).item()
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
        f1_val = f1_score(labels[idx_val].cpu().numpy(), output_preds[idx_val].cpu().numpy())
        parity_val, equality_val = fair_metric(output_preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].cpu().numpy())
        # test
        acc_test = accuracy(output_preds[idx_test], labels[idx_test]).item()
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        f1_test = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())
        groups_acc = group_acc(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())

        acc_val_list.append(acc_val)
        roc_val_list.append(roc_val.item())
        f1_val_list.append(f1_val.item())
        parity_val_list.append(parity_val)
        equality_val_list.append(equality_val)
        fair_list.append(parity_val+equality_val)
        loss_list.append(loss.item())

        acc_test_list.append(acc_test)
        roc_test_list.append(roc_test.item())
        f1_test_list.append(f1_test.item())
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
    args = get_fairgnn_args(parser)
    # set the result path
    base_path = os.path.join('./'+args.dataset, args.model, 
                            'lr'+str(args.lr)+'wd'+str(args.weight_decay),
                            'hidden'+str(args.num_hidden)+'drop'+str(args.dropout),
                            'alpha'+str(args.alpha)+'beta'+str(args.beta), 
                            )
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

    if args.task == 'eva':
        report(args, base_path, excel=True, draw=False)
        toExcel(args, base_path)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))