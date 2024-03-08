import dgl
import time
import tqdm
import ipdb
import pickle
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from multiprocessing import Lock
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz, load_npz
from fairgnn_utils import *
from utils import group_acc
from models import *
from report import *

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


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # print(args)
    # Load data
    # print(args.dataset)
    if args.dataset in ['german','germanA']:
        dataset = 'german'
        sens_attr = "Gender"
        predict_attr = "GoodCustomer"
        label_number = 100
        sens_number = 100
        path = "../dataset/german"
        test_idx = True        
        adj, features, labels, idx_train, idx_val, idx_test,sens, idx_sens_train = fairgnn_load_german(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number)
        if args.dataset == 'germanA':
            file_path = f'../dataset/german/adj_g2.npz'
            adj = load_npz(file_path)

    elif args.dataset in ['credit','creditA']:
        dataset = 'credit'
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        sens_number = 6000
        path_credit = "../dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = fairgnn_load_credit(dataset, sens_attr, 
                                                                                    predict_attr, path=path_credit, 
                                                                                    label_number=label_number, 
                                                                                    sens_number=sens_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        if args.dataset == 'creditA':
            file_path = f'../dataset/credit/adj_1.npz'
            adj = load_npz(file_path)

    elif args.dataset in ['bail','bailA']:
        dataset = 'bail'
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        sens_number = 100
        path_bail = "../dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = fairgnn_load_bail(dataset, sens_attr, 
                                                                                    predict_attr, path=path_bail,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        if args.dataset == 'bailA':
            file_path = f'../dataset/bail/adj_g3.npz'
            adj = load_npz(file_path)

    elif args.dataset == 'nba':
        dataset = 'nba'
        sens_attr = "country"
        sens_idx = 1
        predict_attr = "SALARY"
        label_number = 100
        sens_number = 50
        seed = 20
        path = "../dataset/nba"
        test_idx = True
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = fairgnn_load_pokec(dataset,
                                                                                        sens_attr,
                                                                                        predict_attr,
                                                                                        path=path,
                                                                                        label_number=label_number,
                                                                                        sens_number=sens_number,
                                                                                        seed=seed,test_idx=test_idx)
        features = feature_norm(features)

    elif args.dataset in ['synthetic', 'syn-1', 'syn-2']:
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_syn(args, gen_graph=(args.model!='mlp'))
    
    elif args.dataset == 'sport':
        dataset = 'sport'
        sens_attr='race'
        predict_attr='sport'
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_sport()
    
    elif args.dataset == 'occupation':
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_occ()

    elif args.dataset == 'pokec_z' or 'pokec_n':
        if args.dataset == 'pokec_z':
            dataset = 'region_job'
        else:
            dataset = 'region_job_2'
        sens_attr = "region"
        sens_idx = 1
        predict_attr = "I_am_working_in_field"
        label_number = 500
        sens_number = 200
        seed = 20
        path="../dataset/pokec/"
        test_idx=False
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = fairgnn_load_pokec(dataset,
                                                                                        sens_attr,
                                                                                        predict_attr,
                                                                                        path=path,
                                                                                        label_number=label_number,
                                                                                        sens_number=sens_number,
                                                                                        seed=seed,test_idx=test_idx)
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    labels[labels>1]=1
    G = dgl.DGLGraph()    # dgl 0.4.3
    G.from_scipy_sparse_matrix(adj)
    # G = dgl.from_scipy(adj)
    # G = G.to(torch.device('cuda:0'))

    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Model and optimizer
    model = FairGNN(nfeat = features.shape[1], args = args).to(device)
    if args.cuda:
        model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)
        sens = sens.to(device)
        idx_sens_train = idx_sens_train.to(device)

    from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score

    # Train model
    t_total = time.time()
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
    epoch_list = []
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
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', default=1, help='select gpu.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units of the sensitive attribute estimator')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=4,
                        help='The hyperparameter of alpha')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
    parser.add_argument('--model', type=str, default="GAT",
                        help='the type of model GCN/GAT')
    parser.add_argument('--dataset', type=str, default='pokec_n')
    parser.add_argument('--num-hidden', type=int, default=32,
                        help='Number of hidden units of classifier.')
    parser.add_argument("--num-heads", type=int, default=1,
                            help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--acc', type=float, default=0.688,
                        help='the selected FairGNN accuracy on val would be at least this high')
    parser.add_argument('--roc', type=float, default=0.745,
                        help='the selected FairGNN ROC score on val would be at least this high')
    parser.add_argument('--sens_number', type=int, default=200,
                        help="the number of sensitive attributes")
    parser.add_argument('--label_number', type=int, default=500,
                        help="the number of labels")
    parser.add_argument('--run', type=int, default=0,
                        help="kth run of the model")
    parser.add_argument('--pretrained', type=bool, default=False,
                        help="load a pretrained model")
    parser.add_argument('--task', type=str, default='train', help='train the model or evaluate')
    # synthetic settings
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--dy", type=int, default=24)
    parser.add_argument("--ds", type=int, default=24)
    parser.add_argument("--yscale", type=float, default=0.5)
    parser.add_argument("--sscale", type=float, default=0.5)
    parser.add_argument("--covy", type=int, default=10)
    parser.add_argument("--covs", type=int, default=10)

    args = parser.parse_known_args()[0]

    # base_path = os.path.join('./record', args.model, args.dataset, 
    #                             'lr'+str(args.lr)+'wd'+str(args.weight_decay),
    #                             'hidden'+str(args.num_hidden)+'drop'+str(args.dropout),
    #                             'alpha'+str(args.alpha)+'beta'+str(args.beta), #'1layerw_spectralnorm'
    #                             )
    # if args.dataset == 'synthetic':
    #     base_path = os.path.join('../'+args.dataset+'/balance', f'pysame{args.pysame}ps0same{args.pssame0}ps1same{args.pssame1}pydif{args.pydif}psdif{args.psdif}', 
    #                             args.model,
    #                             'lr'+str(args.lr)+'wd'+str(args.weight_decay),
    #                             'hidden'+str(args.num_hidden)+'drop'+str(args.dropout),
    #                             'alpha'+str(args.alpha)+'beta'+str(args.beta),
    #                             )
    # elif args.dataset in ['german', 'bail', 'credit']:
    #     base_path = os.path.join('../'+args.dataset, 'modify', args.model, 
    #                             'lr'+str(args.lr)+'wd'+str(args.weight_decay), 'drop'+str(args.dropout))
    # elif args.dataset in ['sport','occupation']:
    #     base_path = os.path.join('../'+args.dataset, args.model, 
    #                             'lr'+str(args.lr)+'wd'+str(args.weight_decay),
    #                             'hidden'+str(args.num_hidden)+'drop'+str(args.dropout),
    #                             'alpha'+str(args.alpha)+'beta'+str(args.beta), 
    #                             )
    base_path = os.path.join('../'+args.dataset, args.model, 
                            'lr'+str(args.lr)+'wd'+str(args.weight_decay),
                            'hidden'+str(args.num_hidden)+'drop'+str(args.dropout),
                            'alpha'+str(args.alpha)+'beta'+str(args.beta), 
                            )
                            
    create_directory_safely(base_path)
    if args.task == 'train':
        if os.path.exists(base_path+'/seed'+str(args.seed)+'history.pkl'): 
            print("this result is existing")
            exit() 

    all_acc = []
    all_auc = []
    all_f1 = []
    all_sp = []
    all_eo = []
    t_total = time.time()

    if args.task == 'train':
        history, groups_acc_list = main(args)
        with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'wb') as file:
            pickle.dump(history, file)

    if args.task == 'eva':
        report(args, base_path, excel=True, draw=False)
        toExcel(args, base_path)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))