import os 
import h5py
import pickle
import random
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.utils import dropout_adj, convert
from utils import *
from process_utils import CheckLabel, compute_statistics, add_edges

def read_sport():
    dataset = 'sport'
    path = '../dataset/sport/topk2/'
    train_ratio = 0.6

    with open(os.path.join(path, 'sport_embedding.pkl'), 'rb') as pkl_file:
        data = pickle.load(pkl_file)    #
    data = data.dropna(subset=['sport'])
    data = data.dropna(subset=['race'])
    data = data.drop_duplicates(subset=['user_id'])
    data = data.reset_index(drop=True)
    print(data)
    # df = pd.read_csv(os.path.join(path,"{}_player.csv".format(dataset)))
    # print(df)
    
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
    edges_unordered = np.genfromtxt(os.path.join(path,"sport_edges.txt"), dtype=int)    #print(edges_unordered, type(edges_unordered), edges_unordered.shape)
    filtered_edges = edges_unordered[np.isin(edges_unordered, idx).all(axis=1)]
    np.savetxt(path+'sport_edgesP.txt', filtered_edges, fmt='%d', delimiter='\t')
    flattened_edges = filtered_edges.flatten()
    unique_values, counts = np.unique(flattened_edges, return_counts=True)
    num_unique_values = len(unique_values)
    print(num_unique_values,filtered_edges.shape)
    values_in_unique_not_in_idx = np.setdiff1d(unique_values, idx)
    values_in_idx_not_in_unique = np.setdiff1d(idx, unique_values)
    print("在 unique_values 中存在但在 idx 中不存在的值:", values_in_unique_not_in_idx)
    print("在 idx 中存在但在 unique_values 中不存在的值:", values_in_idx_not_in_unique)
    exit()

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
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train

def read_occ():
    dataset = 'occ'
    path = '../dataset/occupation'
    df_user = pd.read_csv(os.path.join(path,"{}_user.csv".format(dataset)))
    df_user['id'] = df_user['id'].astype(str)
    user = df_user[['id','gender','area']]
    user = user.drop_duplicates()
    user = user.dropna(subset=['gender'])
    user = user.dropna(subset=['area'])
    user = user.reset_index(drop=True)
    #print(user)
    #id_user = np.array(user['id'],dtype=int)


    with open(os.path.join(path, 'occ_tweet_embedding.pkl'), 'rb') as pkl_file:
        data = pickle.load(pkl_file)    
    df = pd.DataFrame(data)

    # 去除包含日期时间字符串的 'user_id' 行
    df['user_id'] = df['user_id'].astype(str)
    df = df[~df['user_id'].str.contains(r'\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \+\d{4} \d{4}', regex=True)]
    df['embeddings'] = df['embeddings'].apply(lambda x: x[0])
    df = df.groupby('user_id')['embeddings'].apply(lambda x: np.mean(x.head(3))).reset_index()
    #print(df)
    #id_emb = np.array(df['user_id'], dtype=int)
    result_df = df.merge(user, left_on='user_id', right_on='id', how='inner')
    result_df = result_df.drop(columns=['id'])
    result_df.to_hdf('../dataset/occupation/processed/occ_embeddings.h5', key='occ', mode='w')
    #result_df.to_csv('../dataset/occupation/processed/occ_embeddings.csv',index=True)
    df = pd.read_hdf('../dataset/occupation/processed/occ_embeddings.h5', key='occ')
    print(df)
    print(type(df['embeddings'][0]))

    idx = np.array(result_df['user_id'], dtype=int)
    print(idx.shape)
    edges_unordered = np.genfromtxt(os.path.join(path,"occ_edges.txt"), dtype=int)    #print(edges_unordered, type(edges_unordered), edges_unordered.shape)
    filtered_edges = edges_unordered[np.isin(edges_unordered, idx).all(axis=1)]
    print(filtered_edges, type(filtered_edges), filtered_edges.shape)
    np.savetxt('../dataset/occupation/processed/occ_edges.txt', filtered_edges, fmt='%d', delimiter='\t')
    # get labels
    labels = result_df['gender'].values.ravel()
    labels = (labels == 'female').astype(int)  #
    print(labels, labels.shape)
    labels = torch.LongTensor(labels)

    idx_map = {j: i for i, j in enumerate(idx)} 
    edges = np.array(list(map(idx_map.get, filtered_edges.flatten())), dtype=int).reshape(filtered_edges.shape)
    print(edges, edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
     
def AnalyseSemisynthetic(dataname):
    if dataname == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = "../dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataname, sens_attr,
                                                                                predict_attr, 
                                                                                path=path_credit,
                                                                                label_number=label_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif dataname == 'german':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        label_number = 100
        path_german = "../dataset/german"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataname, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )

    elif dataname == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "../dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataname, sens_attr, 
                                                                                predict_attr, path=path_bail,
                                                                                label_number=label_number,
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    
    elif dataname == 'nba':
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
    
    elif dataname == 'pokec_z' or 'pokec_n':
        if dataname == 'pokec_z':
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
    
    # s0y0-s0y0 边增加 2500 条
    adj = add_edges(adj, sens, labels, 2500, (0, 0, 0, 0))

    # s1y1-s1y1 边增加 2500 条
    adj = add_edges(adj, sens, labels, 2500, (1, 1, 1, 1))

    # s1y0-s1y0 边增加 1500 条
    adj = add_edges(adj, sens, labels, 1500, (1, 0, 1, 0))

    # print(type(labels), type(adj), type(sens))
    # CheckLabel(labels, idx_train, idx_val, idx_test)
    compute_statistics(sens, labels, adj)

if __name__ == '__main__':
    
    # dataset_name = ['nba','pokec_z','pokec_n']
    # dataset_name = ['bail','credit','german']
    dataset_name = ['german']
    for dataset in dataset_name:
        print('==========dataset name:', dataset)
        AnalyseSemisynthetic(dataset)

    #read_sport()
    #read_occ()

class DataLoader:
    def __init__(self, args, seed=20):
        self.seed = seed
        self.args = args
    def get_features_and_labels(self, sens_attr, predict_attr, preprocess_func=None):
        idx_features_labels = pd.read_csv(os.path.join(self.path, f"{self.name}.csv"))
        print(idx_features_labels)
        self.pp(idx_features_labels)
    def pp(idx_features_labels, head):
        print(idx_features_labels)
        print(head)