import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import random
import yaml
import time
from scipy.sparse import load_npz
from utils import feature_norm

class DataLoader:
    def __init__(self, args, seed=20):
        self.args = args
        self.seed = seed
        np.random.seed(self.seed)

        self.name = args.dataset
        self.config_path = './config_datasets.yaml'
        self.config = self.load_config()
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
            'synthetic': self.load_synthetic,
            'syn-1': self.load_synthetic,
            'syn-2': self.load_synthetic, 
            'sport': self.load_twitter, 
            'occupation': self.load_twitter,
        }
        self.gen_graph = True
        if self.args.model == 'mlp':
            self.gen_graph = False

    def load_config(self):
        with open(self.config_path, 'r') as file:
            try:
                datasets_config = yaml.safe_load(file)
                config = datasets_config['datasets'][self.name]
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
        # print(idx_features_labels[header], type(idx_features_labels[header]))
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        # sens_features = idx_features_labels[sens_attr]
        
        labels = idx_features_labels[predict_attr]
        labels[labels == -1] = 0
        labels[labels > 1] = 1
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        
        # feature normalization for some datasets
        if self.name == 'nba':
            features = feature_norm(features)
        elif self.name in ['bail','credit']:        
            norm_features = feature_norm(features)
            norm_features[:, self.config['sens_idx']] = features[:, self.config['sens_idx']]
            features = norm_features

        return features, labels, idx_features_labels

    def split_data(self, idx_features_labels, sens_number, labels, label_number, test_idx=False):
        idx = np.arange(labels.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}

        # split setting for semi-synthetic datasets in NIFTYGNN
        if self.name in ['german', 'credit', 'bail', 'germanA', 'creditA', 'bailA']:
            label_idx_0 = np.where(labels==0)[0]
            label_idx_1 = np.where(labels==1)[0]
            random.shuffle(label_idx_0)
            random.shuffle(label_idx_1)
            idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
            idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
            idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
            sens = idx_features_labels[self.config['sens_attr']].values.astype(int)
            sens_idx = set(np.where(sens >= 0)[0])
        # split setting for the real-world datasets in FairGNN
        elif self.name in ['pokecz_z', 'pokec_n', 'nba']:
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
        # split setting in our benchmark
        else:
            idx = np.array(idx_features_labels["user_id"], dtype=int)
            idx_map = {j: i for i, j in enumerate(idx)}
            label_idx = np.where(labels>=0)[0]   
            random.shuffle(label_idx)
            n = len(label_idx)
            train_ratio = self.config['train_ratio']
            idx_train = label_idx[:int(n*train_ratio)]
            idx_val = label_idx[int(n*train_ratio): int(n*(1+train_ratio)/2)]
            idx_test = label_idx[int(n*(1+train_ratio)/2):]
            sens = idx_features_labels[self.config['sens_attr']].values    #print(sens, type(sens))
            sens_idx = set(np.where(sens >= 0)[0])
            #idx_test = np.asarray(list(sens_idx & set(idx_test)))
        
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
            edges_unordered = np.genfromtxt(os.path.join(self.path, f'{self.name}_edges.txt')).astype('int')
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                            dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(num_nodes, num_nodes),
                                dtype=np.float32)
            # symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
        # 'dtype' different from 'astype'
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
        elif self.name in ['germanA', 'creditA', 'bailA']:
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
            header.remove(predict_attr)
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

    def load_pokec(self):
        def _preprocess_pokec(idx_features_labels, header, predict_attr, sens_attr):
            header.remove("user_id")
            header.remove(sens_attr)
            header.remove(predict_attr)
            return idx_features_labels, header
        return self._load_data(self.config['sens_attr'], self.config['predict_attr'], 
                               self.config['sens_num'], self.config['label_num'], 
                               preprocess_func=_preprocess_pokec, 
                               test_idx=self.config['test_idx'])

    def load_twitter(self,):
        def _preprocess_twitter(idx_features_labels, header, predict_attr, sens_attr=None):
            header.remove(predict_attr)
            header.remove('user_id')
            header.remove('embeddings')
            return idx_features_labels, header
        return self._load_data(self.config['sens_attr'], self.config['predict_attr'], 
                               self.config['sens_num'], self.config['label_num'], 
                               preprocess_func=_preprocess_twitter)

    def load_synthetic(self, train_ratio=0.6):
        """load syntheic datasets."""
        print('Loading {} dataset from {}'.format(self.name, self.path))
        # generate synthetic dataset
        if self.args.dataset == 'synthetic':
            syn_dataset = SyntheticGenerator(ifSave=True)
            adj, labels, sens, features = syn_dataset.gen_synthetic(self.gen_graph)
        # load the saved synthetic dataset
        else:
            labels = np.loadtxt(f'{self.path}/{self.name}_label.txt', dtype=int)
            sens = np.loadtxt(f'{self.path}/{self.name}_sens.txt', dtype=int)
            features = np.loadtxt(f'{self.path}/{self.name}_feat.csv', delimiter=',')
            edges = np.loadtxt(f'{self.path}/{self.name}_edges.txt', delimiter=',', dtype=int)
            adj_coo = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(features.shape[0], features.shape[0]))
            adj = adj_coo.tocsr()
        n = features.shape[0]
        if self.gen_graph:
            adj = adj + sp.eye(n)
        features = sp.csr_matrix(features, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

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


class SyntheticGenerator:
    def __init__(self, ifSave=True, seed=20):
        config_path = './config_synthetic.yaml'
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.seed = seed
        np.random.seed(self.seed)
        self.save = ifSave

    def gen_synthetic(self, gen_graph=True):
        # Extract values from configuration
        n, yscale, sscale = self.config['n'], self.config['yscale'], self.config['sscale']
        covy, covs, dy, ds = self.config['covy'], self.config['covs'], self.config['dy'], self.config['ds']
        # Group probabilities
        group_probs = {key: self.config[key] for key in ['prob_s0y0', 'prob_s0y1', 'prob_s1y0', 'prob_s1y1']}
        
        y, s = self._assign_group_membership(n, **group_probs)
        ye = self._generate_features(y, yscale, covy, dy, label=True)
        se = self._generate_features(s, sscale, covs, ds, label=False)

        x = np.hstack((ye, se))
        adj = None
        if gen_graph:
            adj = self._generate_graph(n, s, y, self.config['prob_edges'])

        if self.save:
            name = "syn-{:.4f}".format(time.time())
            path = f"./dataset/{name}"
            adj_array = adj.toarray()
            edges = np.transpose(np.nonzero(adj_array))
            os.makedirs(path, exist_ok=True)
            np.savetxt(f'{path}/{name}_edges.txt', edges, delimiter=',', fmt='%d')
            np.savetxt(f'{path}/{name}_label.txt', y, fmt='%d')
            np.savetxt(f'{path}/{name}_sens.txt', s, fmt='%d')
            np.savetxt(f'{path}/{name}_feat.csv', x, delimiter=',')

        return adj, y, s, x

    def _assign_group_membership(self, n, **group_probs):
        group_probabilities = [group_probs['prob_s0y0'], group_probs['prob_s0y1'], 
                               group_probs['prob_s1y0'], group_probs['prob_s1y1']]
        membership = np.random.choice(4, size=n, p=group_probabilities)
        y = membership // 2
        s = membership % 2
        return y, s
    
    def _generate_features(self, attribute, scale, covariance, dimension, label):
        features = []
        cov_matrix = covariance * np.eye(dimension)
        for attr in attribute:
            mean = scale * np.ones(dimension) * (-1 if attr else 1)
            features.append(np.random.multivariate_normal(mean, cov_matrix))
        return np.array(features)
    
    def _generate_graph(self, n, s, y, prob_edges):
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                edge_type = f's{int(s[i])}y{int(y[i])}-s{int(s[j])}y{int(y[j])}'
                if edge_type in prob_edges:
                    adjacency = np.random.rand() < prob_edges[edge_type]
                    adj_matrix[i, j] = adj_matrix[j, i] = adjacency
        return sp.csr_matrix(adj_matrix)





