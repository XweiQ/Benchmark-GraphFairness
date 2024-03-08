#%%
import pickle
import ipdb
import numpy as np
import scipy.sparse as sp
import torch
import os
import random
import pandas as pd
import dgl
from scipy.spatial import distance_matrix


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id[:200]:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map


def fairgnn_load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/", label_number=1000, sens_number=6000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

#    # Normalize MaxBillAmountOverLast6Months
#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
#
#    # Normalize MaxPaymentAmountOverLast6Months
#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
#
#    # Normalize MostRecentBillAmount
#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
#
#    # Normalize MostRecentPaymentAmount
#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
#
#    # Normalize TotalMonthsOverdue
#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
   
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def fairgnn_load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/", label_number=1000, sens_number=100):
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    
    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def fairgnn_load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="../dataset/german/", label_number=1000, sens_number=100):
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
#    # Normalize Age
#    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
#    # Normalize LoanDuration
#    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
#
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
   
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def fairgnn_load_pokec(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


def calculate_joint_probabilities(y, s):
    joint_counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    total_pairs = len(y)

    for y_val, s_val in zip(y, s):
        joint_counts[(y_val, s_val)] += 1

    joint_probabilities = {(k[0], k[1]): v / total_pairs for k, v in joint_counts.items()}
    return joint_probabilities

def print_joint_probability_chart_plain(joint_probabilities):
    # Calculate marginal probabilities
    p_y0 = joint_probabilities[(0, 0)] + joint_probabilities[(0, 1)]
    p_y1 = joint_probabilities[(1, 0)] + joint_probabilities[(1, 1)]
    p_s0 = joint_probabilities[(0, 0)] + joint_probabilities[(1, 0)]
    p_s1 = joint_probabilities[(0, 1)] + joint_probabilities[(1, 1)]

    # Print joint probability chart with marginal probabilities
    print("Joint porbability of y and s: ")
    print("        s=0      s=1     P(y)")
    print("y=0    {:.2f}     {:.2f}     {:.2f}".format(joint_probabilities[(0, 0)], joint_probabilities[(0, 1)], p_y0))
    print("y=1    {:.2f}     {:.2f}     {:.2f}".format(joint_probabilities[(1, 0)], joint_probabilities[(1, 1)], p_y1))
    print("P(s)   {:.2f}     {:.2f}".format(p_s0, p_s1))

def gen_synthetic(args, gen_graph=True, seed=20):
    
    # Parameters for feature generation
    n = args.n              # num samples
    yscale = args.yscale        
    sscale = args.sscale        
    covy = args.covy        # diagonal elements of convarience matrix of y
    covs = args.covs        # diagonal elements of convarience matrix of s
    dy = args.dy            # dim of feature matrix base on label 
    ds = args.ds            # dim of feature matrix base on sens_attr
    py1 = args.py1          # probability of y=1 
    ps1 = args.ps1          # probability of s=1
    
    # Parameters for graph structure
    pysame = args.pysame    # inner link for label group
    pydif = args.pydif      # inter link for label group
    psdif = args.psdif      # inter link for sens_attr group
    pssame0 = args.pssame0  # inner link for s=0
    pssame1 = args.pssame1  # inner link for s=1  (5*pssame0)
    
    # Generate labels y and senstive attr s
    np.random.seed(seed)
    y = np.random.binomial(1, py1, n)
    s = np.random.binomial(1, ps1, n)
    # s_y1 = np.random.binomial(1, 0.75, n)
    # s_y0 = np.random.binomial(1, 0.25, n)
    # s = np.bitwise_and(y, s_y1) + np.bitwise_and(1-y, s_y0)
    
    cov_ye = covy*np.identity(dy)
    cov_se = covs*np.identity(ds)
    # Generate n samples of ye and se with dimension d1 from separate multivariate Gaussian distributions
    ye = []
    se = []
    for yi in y:
        if yi == 1:
            ye.append(np.random.multivariate_normal(-1*yscale*np.ones(dy), cov_ye))
        elif yi == 0:
            ye.append(np.random.multivariate_normal(yscale*np.ones(dy), cov_ye))
    for si in s:
        if si == 1:
            se.append(np.random.multivariate_normal(-1*sscale*np.ones(ds), cov_se))
        elif si == 0:
            se.append(np.random.multivariate_normal(sscale*np.ones(ds), cov_se))
    ye = np.array(ye)
    se = np.array(se)
    # ye *= yscale
    # se *= sscale
    # Concatenate ye and se together for each sample
    x = np.hstack((ye, se))
    print(x.shape)
    # Calculate the joint probabilities from y and s
    joint_probabilities = calculate_joint_probabilities(y, s)
    print_joint_probability_chart_plain(joint_probabilities)
    print(f'correlation coef of y and s: {np.corrcoef(s, y)[0,1]:.4f}')
    exit()
    if not gen_graph:
        return None, y, s, x
    
    # Generate label graph 'adj_y' and sens_attr graph 'adj_s'
    adj_y = y[:, np.newaxis]== y[np.newaxis, :]     # two samples with the same label -> link
    adj_s = s[:, np.newaxis]== s[np.newaxis, :]     # two samples with the same sens_attr -> link
    adj_y = adj_y.astype(int)
    adj_s = adj_s.astype(int)
    adj_s[s.astype(bool)] *= 2                      # different sens_attr,0; two s=0,1; two s=1,2; 
    adj_y = np.array([pydif,pysame])[adj_y]
    # adj_s = np.array([psdif,pssame])[adj_s]
    adj_s = np.array([psdif,pssame0, pssame1])[adj_s]
    adj_y = np.random.rand(n,n)<adj_y
    adj_s = np.random.rand(n,n)<adj_s
    adj_y = adj_y.astype(int)
    adj_s = adj_s.astype(int)
    adj = adj_y|adj_s
    adj = np.triu(adj) + np.triu(adj, 1).T
    adj = sp.csr_matrix(adj)
    
    # draw_graph_stata(adj, y, s)
    return adj, y, s, x

def load_syn(args, gen_graph=True, train_ratio=0.6,seed=20):
    adj, labels, sens, features = gen_synthetic(args, gen_graph)
    n = features.shape[0]

    features = sp.csr_matrix(features, dtype=np.float32)
    if gen_graph:
        adj = adj + sp.eye(n)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    idx_train = idx[:int(n*train_ratio)]
    idx_val = idx[int(n*train_ratio): int(n*(1+train_ratio)/2)]
    idx_test = idx[int(n*(1+train_ratio)/2):]
    sens_idx = set(np.where(sens >= 0)[0])
    #idx_test = np.asarray(list(sens_idx & set(idx_test)))
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)
    
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

def load_sport(path="../dataset/sport/topk", train_ratio=0.6,seed=20,test_idx=False):
    with open(os.path.join(path, 'sport_embedding.pkl'), 'rb') as pkl_file:
        data = pickle.load(pkl_file)    
    data = data.dropna(subset=['sport'])
    data = data.dropna(subset=['race'])
    data = data.drop_duplicates(subset=['user_id'])
    data = data.reset_index(drop=True)
    #print(data)
    # get features
    emb = pd.Series(data['embeddings']).apply(lambda x: x[0])
    emb = pd.DataFrame(emb.tolist())    #print(emb, type(emb), emb.shape)
    features = sp.csr_matrix(emb.values, dtype=np.float32)  #print(features, type(features))
    features = torch.FloatTensor(np.array(features.todense()))
    # get labels
    labels = data['sport'].values.ravel()
    labels = (labels == 'mlb').astype(int)  #print(labels, type(labels))
    labels = torch.LongTensor(labels)
    # build graph
    idx = np.array(data['user_id'], dtype=int)  #print(idx, type(idx), idx.shape)
    idx_map = {j: i for i, j in enumerate(idx)} 
    edges_unordered = np.genfromtxt(os.path.join(path,"sport_edgesP.txt"), dtype=int)    #print(edges_unordered, type(edges_unordered), edges_unordered.shape)
    filtered_edges = edges_unordered[np.isin(edges_unordered, idx).all(axis=1)]
    edges = np.array(list(map(idx_map.get, filtered_edges.flatten())), dtype=int).reshape(filtered_edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    # get sens & idx
    sens = np.array(data['race'], dtype=int)    #print(sens, type(sens))
    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    random.seed(20)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)
    n = len(label_idx)
    idx_train = label_idx[:int(n*train_ratio)]
    idx_val = label_idx[int(n*train_ratio): int(n*(1+train_ratio)/2)]
    idx_test = label_idx[int(n*(1+train_ratio)/2):]
    #idx_test = np.asarray(list(sens_idx & set(idx_test)))
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train

def load_occ(path='../dataset/occupation/processed', train_ratio=0.6):
    dataset = 'occ'

    data = pd.read_hdf(os.path.join(path,'{}_embeddings.h5'.format(dataset)), key='occ')  #print(data)
    # get features
    emb = pd.DataFrame(data['embeddings'].tolist())    #print(emb, type(emb), emb.shape)
    features = sp.csr_matrix(emb.values, dtype=np.float32)  #print(features, type(features))
    features = torch.FloatTensor(np.array(features.todense()))
    # get labels
    labels = np.array(data['area'], dtype=int)    #print(labels, type(labels))
    labels = torch.LongTensor(labels)
    idx = np.array(data['user_id'], dtype=int)  #print(idx, type(idx), idx.shape)
    idx_map = {j: i for i, j in enumerate(idx)} 
    edges_unordered = np.genfromtxt(os.path.join(path,'{}_edges.txt'.format(dataset)), dtype=int)    #print(edges_unordered, type(edges_unordered), edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    # get sens & idx
    sens = data['gender'].values.ravel()
    sens = (sens == 'female').astype(int)    #print(sens, type(sens))
    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    random.seed(20)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)
    n = len(label_idx)
    idx_train = label_idx[:int(n*train_ratio)]
    idx_val = label_idx[int(n*train_ratio): int(n*(1+train_ratio)/2)]
    idx_test = label_idx[int(n*(1+train_ratio)/2):]
    #idx_test = np.asarray(list(sens_idx & set(idx_test)))
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
