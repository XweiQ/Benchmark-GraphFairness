import os
import dgl
import torch
import random
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Lock
from collections import Counter
    
def create_directory_safely(base_path):
    """
    Avoiding conflicts during multithread-training.
    """
    lock = Lock()
    with lock:
        if not os.path.exists(base_path):
            os.makedirs(base_path,exist_ok=True)

def compute_statistics(sens, labels, adj, x):
    num_nodes, num_attr = x.shape[0], x.shape[1]
    print(f'nodes: {num_nodes}, attr: {num_attr}')
    # Counting number and proportion of nodes in the four type groups
    node_counts = Counter()
    for s, y in zip(sens, labels):
        node_counts[(s.item(), y.item())] += 1

    total_nodes = len(sens)
    node_proportions = {k: v / total_nodes for k, v in node_counts.items()}
    node_types = ['S=0 Y=0', 'S=0 Y=1', 'S=1 Y=0', 'S=1 Y=1']

    node_counts_list = [int(node_counts.get((s, y), 0)) for s in [0, 1] for y in [0, 1]]
    node_proportions_list = [round(node_proportions.get((s, y), 0), 4) for s in [0, 1] for y in [0, 1]]
    df_nodes = pd.DataFrame([node_counts_list, node_proportions_list], columns=node_types, index=['数量', '比例'])
    print(df_nodes)

    # Counting number and proportion of ten undirected edges
    edge_counts = Counter()
    for i, j in zip(*adj.nonzero()):
        if i < j:
            edge_type = (sens[i].item(), labels[i].item(), sens[j].item(), labels[j].item())
            edge_counts[edge_type] += 1

    total_edges = sum(edge_counts.values())
    edge_proportions = {k: v / total_edges for k, v in edge_counts.items()}

    edge_types = [
        's0y0-s0y0', 's0y1-s0y1', 's1y0-s1y0', 's1y1-s1y1', # s same y same
        's0y0-s1y0', 's0y1-s1y1', # s diff y same
        's0y0-s0y1', 's1y0-s1y1', # s same y diff
        's0y0-s1y1', 's0y1-s1y0'  # s diff y diff
    ]

    edge_counts_list = [0] * 10
    edge_proportions_list = [0] * 10

    for ((s1, y1, s2, y2), count) in edge_counts.items():
        edge_type = tuple(sorted([(s1, y1), (s2, y2)]))
        edge_type_str = f's{int(edge_type[0][0])}y{int(edge_type[0][1])}-s{int(edge_type[1][0])}y{int(edge_type[1][1])}'

        if edge_type_str in edge_types:
            index = edge_types.index(edge_type_str)
            edge_counts_list[index] += count
            edge_proportions_list[index] += edge_proportions[(s1, y1, s2, y2)]

    edge_counts_list = [int(count) for count in edge_counts_list]
    edge_proportions_list = [round(proportion, 4) for proportion in edge_proportions_list]

    df_edges = pd.DataFrame([edge_counts_list, edge_proportions_list], columns=edge_types, index=['数量', '比例'])
    print('num_edges: ', total_edges)
    print(df_edges)

    # Calculate the distribution of node degrees
    degrees = adj.sum(axis=1).A.flatten()
    degree_distribution = Counter(degrees)
    average_degree = np.mean(degrees)
    print('avg degree: ', average_degree)

    return node_counts, node_proportions, edge_counts, edge_proportions, degree_distribution

def add_edges(adj, sens, labels, num_edges_to_add, edge_type):
    np.random.seed(42)
    
    # Filtering eligible nodes
    nodes_a = [i for i in range(len(sens)) if sens[i] == edge_type[0] and labels[i] == edge_type[1]]
    nodes_b = [i for i in range(len(sens)) if sens[i] == edge_type[2] and labels[i] == edge_type[3]]

    potential_edges = set((a, b) for a in nodes_a for b in nodes_b if a < b)

    # Removing pre-existing edges from potential edges
    adj_coo = adj.tocoo()
    existing_edges = set(zip(adj_coo.row, adj_coo.col))
    potential_edges -= existing_edges

    # Convert the set of potential edges into a list and randomly select the edges to be added
    potential_edges_list = list(potential_edges)
    num_edges_to_add = min(num_edges_to_add, len(potential_edges_list))
    selected_indices = np.random.choice(len(potential_edges_list), size=num_edges_to_add, replace=False)
    new_edges = [potential_edges_list[i] for i in selected_indices]

    # add edges
    rows, cols = zip(*new_edges)
    data = np.ones(len(rows))
    new_edges_matrix = sp.coo_matrix((data, (rows, cols)), shape=adj.shape)

    # Combining old and new adjacency matrices
    adj = adj + new_edges_matrix + new_edges_matrix.T  

    return adj.tocsr()

def remove_edges(adj, sens, labels, num_edges_to_remove, edge_type):
    np.random.seed(42)

    adj_coo = adj.tocoo()
    possible_edges = [(i, j) for i, j in zip(adj_coo.row, adj_coo.col) if
                      ((sens[i], labels[i], sens[j], labels[j]) == edge_type or 
                       (sens[j], labels[j], sens[i], labels[i]) == edge_type) and i < j]

    # Randomly select the index of the edge to be removed
    num_edges_to_remove = min(num_edges_to_remove, len(possible_edges))
    remove_indices = np.random.choice(len(possible_edges), size=num_edges_to_remove, replace=False)

    # Construct an array to indicate which edges need to be preserved
    keep_mask = np.ones(len(adj_coo.data), dtype=bool)
    for idx in remove_indices:
        i, j = possible_edges[idx]
        keep_mask &= ~((adj_coo.row == i) & (adj_coo.col == j))
        keep_mask &= ~((adj_coo.row == j) & (adj_coo.col == i))

    # Using Boolean indexes to keep un-removed edges
    adj_coo.data = adj_coo.data[keep_mask]
    adj_coo.row = adj_coo.row[keep_mask]
    adj_coo.col = adj_coo.col[keep_mask]

    # Reconstructing the sparse matrix
    adj_new = sp.coo_matrix((adj_coo.data, (adj_coo.row, adj_coo.col)), shape=adj.shape).tocsr()

    return adj_new

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

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def group_acc(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y0 = np.bitwise_and(idx_s0, labels==0)
    idx_s1_y0 = np.bitwise_and(idx_s1, labels==0)
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    acc_s0_y0 = sum(np.logical_not(pred[idx_s0_y0]))/sum(idx_s0_y0)
    acc_s1_y0 = sum(np.logical_not(pred[idx_s1_y0]))/sum(idx_s1_y0)
    acc_s0_y1 = sum(pred[idx_s0_y1])/sum(idx_s0_y1)
    acc_s1_y1 = sum(pred[idx_s1_y1])/sum(idx_s1_y1)
    return [acc_s0_y0, acc_s0_y1, acc_s1_y0, acc_s1_y1]



if __name__ == '__main__':
    
    print("finish.")
