#%%
import dgl
import ipdb
import time
import pickle
import argparse
import numpy as np
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import warnings
# from baseline.nifty.utils import *
# from baseline.nifty.models import *
warnings.filterwarnings('ignore')
from multiprocessing import Lock
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
from scipy.sparse import save_npz, load_npz
from report import *
from utils import *
from models import *
from process_utils import *

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def ssf_validation(model, idx_val, x_1, edge_index_1, x_2, edge_index_2, y, device):
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = args.sim_coeff*(l1+l2)

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4

def main(args):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # Load data
    # print(args.dataset)

    # Load credit_scoring dataset
    if args.dataset in ['credit','creditA']:
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = "../dataset/credit"
        dataset = 'credit'
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_attr,
                                                                                predict_attr, path=path_credit,
                                                                                label_number=label_number
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

        if args.dataset == 'creditA':
            file_path = f'../dataset/credit/adj_1.npz'
            adj = load_npz(file_path)

    elif args.dataset in ['synthetic', 'syn-1', 'syn-2']:
        dataset = 'synthetic'
        sens_idx = 1
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_syn(args, gen_graph=(args.model!='mlp'))
    
    elif args.dataset == 'sport':
        dataset = 'sport'
        sens_attr='race'
        sens_idx = 1
        predict_attr='sport'
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_sport()
    
    elif args.dataset == 'occupation':
        sens_idx = 1
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_occ()

    elif args.dataset in ['german','germanA']:
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        label_number = 100
        path_german = "../dataset/german"
        dataset = 'german'
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )
        if args.dataset == 'germanA':
            file_path = f'../dataset/german/adj_g2.npz'
            adj = load_npz(file_path)
    # Load bail dataset
    elif args.dataset in ['bail','bailA']:
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "../dataset/bail"
        dataset = 'bail'
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_attr, 
                                                                                predict_attr, path=path_bail,
                                                                                label_number=label_number,
                                                                                )
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
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                        sens_attr,
                                                                                        predict_attr,
                                                                                        path=path,
                                                                                        label_number=label_number,
                                                                                        sens_number=sens_number,
                                                                                        seed=seed,test_idx=test_idx)
        features = feature_norm(features)
    
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
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                        sens_attr,
                                                                                        predict_attr,
                                                                                        path=path,
                                                                                        label_number=label_number,
                                                                                        sens_number=sens_number,
                                                                                        seed=seed,test_idx=test_idx)
    else:
        print('Invalid dataset name!!')
        exit(0)
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    # print("adj", type(adj))
    # print("features", features.size())
    # print(labels.size(), idx_train.size(), idx_val.size(), idx_test.size(),sens.size())
    # exit()

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Model and optimizer
    labels[labels>1]=1
    num_class = labels.unique().shape[0]-1
    if args.dataset == 'nba' or 'pokec_z' or 'pokec_n':
        num_class = 1
    # print(labels, num_class)
    if args.model == 'gcn':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nlayers=args.num_layers,
                    nclass=num_class,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model == 'sage':
        model = SAGE(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=num_class,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model=="mlp":
        model = MLP(features.shape[1],
                    args.hidden,
                    num_class,
                    args.num_layers)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model == 'gin':
        model = GIN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=num_class,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model == 'jk':
        model = JK(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=num_class,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model == 'infomax':
        enc_dgi = Encoder_DGI(nfeat=features.shape[1], nhid=args.hidden)
        enc_cls = Encoder_CLS(nhid=args.hidden, nclass=num_class)
        model = GraphInfoMax(enc_dgi=enc_dgi, enc_cls=enc_cls)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)

    elif args.model == 'rogcn':
        model = RobustGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nhid=args.hidden, nclass=num_class, dropout=args.dropout, device=device, seed=args.seed)

    elif args.model == 'ssf':
        encoder = Encoder(in_channels=features.shape[1], out_channels=args.hidden, base_model=args.encoder).to(device)	
        model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff, nclass=num_class).to(device)
        val_edge_index_1 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_1)[0]
        val_edge_index_2 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_2)[0]
        val_x_1 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, sens_flag=False)
        val_x_2 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx)
        par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
        par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
        optimizer_1 = optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_2 = optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)


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
    if args.model == 'rogcn':
        model.fit(features, adj, labels, idx_train, idx_val=idx_val, idx_test=idx_test, verbose=True, attention=False, train_iters=args.epochs)

    for epoch in range(args.epochs+1):
        t = time.time()

        if args.model in ['mlp']:
            model.train()
            optimizer.zero_grad()
            output = model(features)

            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(labels)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

            auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately,
            model.eval()
            output = model(features)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
            loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))

            acc_val = accuracy(output[idx_val], labels[idx_val])
            auc_roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), preds[idx_val].detach().cpu().numpy())
            f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())

            acc_val_list.append(acc_val.item())
            roc_val_list.append(auc_roc_val.item())
            f1_val_list.append(f1_val.item())
            parity_val_list.append(parity_val)
            equality_val_list.append(equality_val)
            fair_list.append(parity_val+equality_val)

            # test
            output_preds = (output.squeeze()>0).type_as(labels)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            # if epoch % 100 == 0:
            #     print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

            acc_test_list.append(acc_test.item())
            roc_test_list.append(auc_roc_test.item())
            f1_test_list.append(f1_s.item())
            parity_list.append(parity)
            equality_list.append(equality)

            # different early stop
            if auc_roc_val > args.roc:
                if start_fair == 0:
                    start_epoch = epoch
                    start_fair = 1
                if best_fair > parity_val + equality_val :
                    best_fair = parity_val + equality_val 
                    stop_epoch = epoch
                    torch.save(model.state_dict(), f'./weights/weights_vanilla_{args.model}_{args.num_layers}_{args.dataset}.pt')

        elif args.model in ['gcn', 'sage', 'gin', 'jk', 'infomax']:
            model.train()
            optimizer.zero_grad()
            output = model(features, edge_index)

            # Binary Cross-Entropy  
            preds = (output.squeeze()>0).type_as(labels)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

            auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately,
            model.eval()
            output = model(features, edge_index)

            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
            loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))

            acc_val = accuracy(output[idx_val], labels[idx_val])
            auc_roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), preds[idx_val].detach().cpu().numpy())
            f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())

            acc_val_list.append(acc_val.item())
            roc_val_list.append(auc_roc_val.item())
            f1_val_list.append(f1_val.item())
            parity_val_list.append(parity_val)
            equality_val_list.append(equality_val)
            fair_list.append(parity_val+equality_val)

            # test
            output_preds = (output.squeeze()>0).type_as(labels)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            # if epoch % 100 == 0:
            #     print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

            acc_test_list.append(acc_test.item())
            roc_test_list.append(auc_roc_test.item())
            f1_test_list.append(f1_s.item())
            parity_list.append(parity)
            equality_list.append(equality)

            # different early stop
            if auc_roc_val > args.roc:
                if start_fair == 0:
                    start_epoch = epoch
                    start_fair = 1
                if best_fair > parity_val + equality_val :
                    best_fair = parity_val + equality_val 
                    stop_epoch = epoch
                    torch.save(model.state_dict(), f'./weights/weights_vanilla_{args.model}_{args.num_layers}_{args.dataset}.pt')
        
        elif args.model == 'ssf':
            sim_loss = 0
            cl_loss = 0
            rep = 1
            for _ in range(rep):
                model.train()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
                x_1 = drop_feature(features, args.drop_feature_rate_2, sens_idx, sens_flag=False)
                x_2 = drop_feature(features, args.drop_feature_rate_2, sens_idx)
                z1 = model(x_1, edge_index_1)
                z2 = model(x_2, edge_index_2)

                # projector
                p1 = model.projection(z1)
                p2 = model.projection(z2)

                # predictor
                h1 = model.prediction(p1)
                h2 = model.prediction(p2)

                l1 = model.D(h1[idx_train], p2[idx_train])/2
                l2 = model.D(h2[idx_train], p1[idx_train])/2
                sim_loss += args.sim_coeff*(l1+l2)

            (sim_loss/rep).backward()
            optimizer_1.step()

            # classifier
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)
            c1 = model.classifier(z1)
            c2 = model.classifier(z2)

            # Binary Cross-Entropy   
            l3 = F.binary_cross_entropy_with_logits(c1[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2
            l4 = F.binary_cross_entropy_with_logits(c2[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2

            cl_loss = (1-args.sim_coeff)*(l3+l4)
            cl_loss.backward()
            optimizer_2.step()
            loss = (sim_loss/rep + cl_loss)

            # Validation
            model.eval()
            val_s_loss, val_c_loss = ssf_validation(model, idx_val, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2, labels, device)
            emb = model(val_x_1, val_edge_index_1)
            output = model.predict(emb)
            preds = (output.squeeze()>0).type_as(labels)
            acc_val = accuracy(preds[idx_val], labels[idx_val])
            auc_roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
            f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())

            acc_val_list.append(acc_val.item())
            roc_val_list.append(auc_roc_val.item())
            f1_val_list.append(f1_val.item())
            parity_val_list.append(parity_val)
            equality_val_list.append(equality_val)
            fair_list.append(parity_val+equality_val)
            loss_list.append((val_c_loss + val_s_loss).item())

            # test
            emb = model(features.to(device), edge_index.to(device))
            output = model.predict(emb)
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
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', default=0, help='select gpu.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    # model setting            
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,
                        help='drop edge for first augmented graph')
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                        help='drop edge for second augmented graph')
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.1,
                        help='drop feature for first augmented graph')
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.1,
                        help='drop feature for second augmented graph')
    parser.add_argument('--sim_coeff', type=float, default=0.5,
                        help='regularization coeff for the self-supervised task')
    parser.add_argument('--dataset', type=str, default='occupation')
    parser.add_argument("--num_heads", type=int, default=1,
                            help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                            help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn', 'mlp'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--roc', type=float, default=0.60, help='the selected ROC score on val would be at least this high')
    parser.add_argument('--acc', type=float, default=0.688, help='the selected accuracy on val would be at least this high')
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

    if args.dataset == 'synthetic':
        base_path = os.path.join('../'+args.dataset+'/balance', f'pysame{args.pysame}ps0same{args.pssame0}ps1same{args.pssame1}pydif{args.pydif}psdif{args.psdif}', 
                                    args.model+str(args.num_layers), 
                                'lr'+str(args.lr)+'wd'+str(args.weight_decay), 'drop'+str(args.dropout))
    elif args.dataset in ['german', 'bail', 'credit']:
        base_path = os.path.join('../'+args.dataset, 'modify', args.model+'_'+args.encoder, 
                                'lr'+str(args.lr)+'wd'+str(args.weight_decay),
                                'drop'+str(args.dropout), 'coeff'+str(args.sim_coeff))    
    elif args.dataset in ['sport','occupation']:
        base_path = os.path.join('../'+args.dataset, args.model+'_'+args.encoder, 
                                'lr'+str(args.lr)+'wd'+str(args.weight_decay),
                                'drop'+str(args.dropout), 'coeff'+str(args.sim_coeff))
    base_path = os.path.join('../'+args.dataset, "OriginalSplit",  
                                args.model+'_'+args.encoder, 
                                'lr'+str(args.lr)+'wd'+str(args.weight_decay),
                                'drop'+str(args.dropout), 'coeff'+str(args.sim_coeff))
    
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
        history, group_acc_list = main(args)
        with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'wb') as file:
            pickle.dump(history, file)
        evaluate(args, base_path, group_acc_list)

    if args.task == 'eva':
        report(args, base_path, excel=True, draw=False)
        toExcel(args, base_path)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

