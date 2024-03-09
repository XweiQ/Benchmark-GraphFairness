#%%
import dgl
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
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert

from get_args import *
from load_data import DataLoader, SyntheticGenerator
from models import *
from report import *
from utils import *

# utils functions for niftygnn.
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
    if args.model == 'ssf':
        sens_idx = data_loader.config['sens_idx']
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

        if args.model == 'ssf':
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
    # get arguments
    parser = argparse.ArgumentParser()
    args = get_nifty_args(parser)
    # set the result path
    base_path = os.path.join('./'+args.dataset,   
                                args.model+'_'+args.encoder, 
                                'lr'+str(args.lr)+'wd'+str(args.weight_decay),
                                'drop'+str(args.dropout), 'coeff'+str(args.sim_coeff))
    create_directory_safely(base_path)
    
    t_total = time.time()
    
    if args.task == 'train':
        if os.path.exists(base_path+'/seed'+str(args.seed)+'history.pkl'): 
            print("this result is existing") 
            exit()
        else:
            history, group_acc_list = main(args)
            with open(base_path+'/seed'+str(args.seed)+'history.pkl', 'wb') as file:
                pickle.dump(history, file)
            evaluate(args, base_path, group_acc_list)
    if args.task == 'eva':
        report(args, base_path, excel=True, draw=False)
        toExcel(args, base_path)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

