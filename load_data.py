import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import random
import yaml
import pickle
from scipy.sparse import load_npz

class DataLoader:
    def __init__(self, args, seed=20):
        self.seed = seed
        self.args = args

        self.name = args.dataset
        self.config = self.load_config(self.name)
        self.path = f'./dataset/{self.name}'
        self.map = {
            'bail': self.load_bail,
            'bailA': self.load_bailA,
            'credit': self.load_credit,
            'creditA': self.load_creditA,
            'german': self.load_german,
            'germanA': self.load_germanA,
            'nba': self.load_pokec,
            'pokec_n': self.load_pokec,
            'pokec_z': self.load_pokec,
            #'synthetic': self.load_synthetic,
            #'syn-1': self.load_synthetic,
            #'syn-2': self.load_synthetic, 
            #'sport': self.load_sport, 
            #'occupation': self.load_occupation
        }

    def load_config(self, name):
        with open('datasets_config.yaml', 'r') as file:
            try:
                datasets_config = yaml.safe_load(file)
                config = datasets_config['datasets'][name]
                return config
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def load_dataset(self):
        # Call the appropriate data load function based on args.dataset
        if self.args.dataset in self.map:
            return self.map[self.name]()
        else:
            raise ValueError(f"Dataset {self.name} not recognized")
        
    def _load_data(self, sens_attr, predict_attr, sens_number, label_number, preprocess_func=None, test_idx=False):
        print('Loading {} dataset from {}'.format(self.name, self.path))
        
        features, labels, idx_features_labels = self.get_features_and_labels(sens_attr, predict_attr, preprocess_func)

        idx_train, idx_val, idx_test, idx_map, sens, idx_sens_train = self.split_data(idx_features_labels, sens_number, labels, label_number)

        adj = self.build_adjacency(labels.shape[0], idx_map)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

    def get_features_and_labels(self, sens_attr, predict_attr, preprocess_func=None):
        idx_features_labels = pd.read_csv(os.path.join(self.path, f"{self.name}.csv"))
        header = list(idx_features_labels.columns)
        if preprocess_func:
            idx_features_labels, header = preprocess_func(idx_features_labels, header, predict_attr, sens_attr)
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        # sens_features = idx_features_labels[sens_attr]
        
        labels = idx_features_labels[predict_attr]
        labels[labels == -1] = 0
        labels[labels > 1] = 1
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        
        return features, labels, idx_features_labels

    def split_data(self, idx_features_labels, sens_number, labels, label_number, test_idx=False):
        idx = np.arange(labels.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        random.seed(self.seed)

        if self.name in ['pokecz_z', 'pokec_n', 'nba']:
            idx = np.array(idx_features_labels["user_id"], dtype=int)   # print(idx)
            idx_map = {j: i for i, j in enumerate(idx)}        
            label_idx = np.where(labels>=0)[0]
            random.shuffle(label_idx)
            idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
            idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
            if test_idx:
                idx_test = label_idx[label_number:]
                idx_val = idx_test
            else:
                idx_test = label_idx[int(0.75 * len(label_idx)):]
            sens = idx_features_labels[self.config['sens_attr']].values
            sens_idx = set(np.where(sens >= 0)[0])
            idx_test = np.asarray(list(sens_idx & set(idx_test)))
        elif self.name in ['german', 'credit', 'bail', 'germanA', 'creditA', 'bailA']:
            label_idx_0 = np.where(labels==0)[0]
            label_idx_1 = np.where(labels==1)[0]
            random.shuffle(label_idx_0)
            random.shuffle(label_idx_1)
            idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
            idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
            idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
            sens = idx_features_labels[self.config['sens_attr']].values.astype(int)
            sens_idx = set(np.where(sens >= 0)[0])
        
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.shuffle(idx_sens_train)    
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return idx_train, idx_val, idx_test, idx_map, sens, idx_sens_train

    def build_adjacency(self, num_nodes, idx_map):
        # build adj from edges (.txt files)
        if self.name in ['german', 'credit', 'bail']:
            # 'astype' different from 'dtype'
            edges_unordered = np.genfromtxt(os.path.join(self.path, f'{self.name}_edges.txt')).astype('int')
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                            dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(num_nodes, num_nodes),
                                dtype=np.float32)
            # symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
        elif self.name in ['nba', 'pokec_n', 'pokec_z' ,'sport', 'occupation']:
            edges_unordered = np.genfromtxt(os.path.join(self.path, f'{self.name}_edges.txt'), dtype=int)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                            dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(num_nodes, num_nodes),
                                dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
        # load adj from adjacency matrix (.npz files)
        elif self.name in ['germanA', 'creditA', 'bailA', 'syn-1', 'syn-2']:
            adj = load_npz(f'{self.path}/{self.name}_edges.npz')
        else:
            raise ValueError(f"No adjacency build function for this dataset: {self.name}.")
        
        return adj

    def load_credit(self):
        def _preprocess_credit(idx_features_labels, header, predict_attr, sens_attr=None):
            header.remove(predict_attr)
            header.remove('Single')    
            return idx_features_labels, header
        return self._load_data(self.config['sens_attr'], self.config['predict_attr'], 
                               self.config['sens_num'], self.config['label_num'], 
                               preprocess_func=_preprocess_credit)

    def load_bail(self):
        def _preprocess_bail(idx_features_labels, header, predict_attr, sens_attr=None):
            header.remove(predict_attr)
            return idx_features_labels, header
        return self._load_data(self.config['sens_attr'], self.config['predict_attr'], 
                               self.config['sens_num'], self.config['label_num'], 
                               preprocess_func=_preprocess_bail)

    def load_german(self):
        def _preprocess_german(idx_features_labels, header, predict_attr, sens_attr=None):
            header.remove('OtherLoansAtStore')
            header.remove('PurposeOfLoan')
            idx_features_labels.loc[idx_features_labels['Gender'] == 'Female', 'Gender'] = 1
            idx_features_labels.loc[idx_features_labels['Gender'] == 'Male', 'Gender'] = 0
            return idx_features_labels, header
        return self._load_data(self.config['sens_attr'], self.config['predict_attr'], 
                               self.config['sens_num'], self.config['label_num'], 
                               preprocess_func=_preprocess_german)

    def load_creditA(self):
        return self.load_credit()

    def load_bailA(self):
        return self.load_bail()

    def load_germanA(self):
        return self.load_german()

    def _preprocess_pokec(self, idx_features_labels, header, predict_attr, sens_attr):
        header.remove("user_id")
        header.remove(sens_attr)
        header.remove(predict_attr)
        return idx_features_labels, header
    
    def load_pokec(self):
        return self._load_data(self.config['sens_attr'], self.config['predict_attr'], 
                               self.config['sens_num'], self.config['label_num'], 
                               preprocess_func=self._preprocess_pokec, 
                               test_idx=self.config['test_idx'])

    def load_twitter(self,):


class DataLoader_copy:
    def __init__(self, args):
        self.seed = 20
        self.args = args
        self.name = args.dataset
        self.path = f'./dataset/{self.name}'
        self.map = {
            'bail': self.load_bail,
            'bailA': self.load_bailA,
            'credit': self.load_credit,
            'creditA': self.load_creditA,
            'german': self.load_german,
            'germanA': self.load_germanA,
            #'nba': self.load_pokec,
            #'pokec_n': self.load_pokec,
            #'pokec_z': self.load_pokec,
            #'synthetic': self.load_synthetic,
            #'syn-1': self.load_synthetic,
            #'syn-2': self.load_synthetic, 
            #'sport': self.load_sport, 
            #'occupation': self.load_occupation
        }

    def load_dataset(self):
        # Call the appropriate data load function based on args.dataset
        if self.args.dataset in self.map:
            return self.map[self.name]()
        else:
            raise ValueError(f"Dataset {self.name} not recognized")
        
    def _load_data(self, sens_attr, predict_attr, label_number, preprocess_func=None):
        print('Loading {} dataset from {}'.format(self.name, self.path))
        
        features, labels, sens_features = self.get_features_and_labels(sens_attr, predict_attr, preprocess_func)
        # get features and labels
        # idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
        # header = list(idx_features_labels.columns)
        # header.remove(predict_attr)
        # if preprocess_func:
        #     idx_features_labels, header = preprocess_func(idx_features_labels, header)
        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        # labels = idx_features_labels[predict_attr].values
        # labels[labels == -1] = 0
        # labels[labels > 1] = 1
        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(labels)
        
        idx_train, idx_val, idx_test, idx_map, sens, idx_sens_train = self.split_data(labels, label_number, sens_features)
        # data split
        # idx = np.arange(features.shape[0])
        # idx_map = {j: i for i, j in enumerate(idx)}

        # import random
        # random.seed(self.seed)
        # label_idx_0 = np.where(labels==0)[0]
        # label_idx_1 = np.where(labels==1)[0]
        # random.shuffle(label_idx_0)
        # random.shuffle(label_idx_1)
        # idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
        # idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        # idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        
        # sens = idx_features_labels[sens_attr].values.astype(int)
        # sens_idx = set(np.where(sens >= 0)[0])
        # sens = torch.FloatTensor(sens)
        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)       
        # idx_sens_train = None

        adj = self.build_adjacency(labels.shape[0], idx_map)
        # build relationship
        # edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
        # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
        #                 dtype=int).reshape(edges_unordered.shape)
        # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                     shape=(labels.shape[0], labels.shape[0]),
        #                     dtype=np.float32)
        # # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj + sp.eye(adj.shape[0])

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

    def get_features_and_labels(self, sens_attr, predict_attr, preprocess_func=None):
        idx_features_labels = pd.read_csv(os.path.join(self.path, f"{self.name}.csv"))
        
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)        
        if preprocess_func:
            idx_features_labels, header = preprocess_func(idx_features_labels, header)
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        sens_features = idx_features_labels[sens_attr]
        
        labels = idx_features_labels[predict_attr]
        labels[labels == -1] = 0
        labels[labels > 1] = 1
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        
        return features, labels, sens_features

    def split_data(self, labels, label_number, sens_features):
        idx = np.arange(labels.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        
        label_idx_0 = np.where(labels==0)[0]
        label_idx_1 = np.where(labels==1)[0]
        np.random.seed(self.seed)
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        sens = sens_features.values.astype(int)
        sens_idx = set(np.where(sens >= 0)[0])
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.shuffle(idx_sens_train)
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(idx_sens_train[:label_number])

        return idx_train, idx_val, idx_test, idx_map, sens, idx_sens_train

    def build_adjacency(self, num_nodes, idx_map):
        edges_unordered = np.genfromtxt(f'{self.path}/{self.name}_edges.txt').astype('int')
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(num_nodes, num_nodes),
                            dtype=np.float32)
        # symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        
        return adj

    def load_credit(self):
        def _preprocess_credit(idx_features_labels, header):
            header.remove('Single')
            return idx_features_labels, header
        return self._load_data("Age", "NoDefaultNextMonth", 6000, preprocess_func=_preprocess_credit)

    def load_bail(self):
        return self._load_data("WHITE", "RECID", 100)

    def load_german(self):
        def _preprocess_german(idx_features_labels, header):
            header.remove('OtherLoansAtStore')
            header.remove('PurposeOfLoan')
            idx_features_labels.loc[idx_features_labels['Gender'] == 'Female', 'Gender'] = 1
            idx_features_labels.loc[idx_features_labels['Gender'] == 'Male', 'Gender'] = 0
            return idx_features_labels, header
        return self._load_data("Gender", "GoodCustomer", 100, preprocess_func=_preprocess_german)
    
    def load_creditA(self):
        dataset = self.load_credit()
        # load the modified adjacency matrix
        dataset[0] = load_npz(f'{self.path}/{self.name}_edges.npz')
        return dataset

    def load_bailA(self):
        dataset = self.load_bail()
        # load the modified adjacency matrix
        dataset[0] = load_npz(f'{self.path}/{self.name}_edges.npz')
        return dataset

    def load_germanA(self):
        dataset = self.load_german()
        # load the modified adjacency matrix
        dataset[0] = load_npz(f'{self.path}/{self.name}_edges.npz')
        return dataset
    
    


