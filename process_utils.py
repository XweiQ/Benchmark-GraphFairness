import numpy as np
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import lil_matrix
from collections import Counter

def compute_statistics(sens, labels, adj, x):
    num_nodes, num_attr = x.shape[0], x.shape[1]
    print(f'nodes: {num_nodes}, attr: {num_attr}')
    # 统计四种类型的节点数量及比例
    node_counts = Counter()
    for s, y in zip(sens, labels):
        node_counts[(s.item(), y.item())] += 1

    total_nodes = len(sens)
    node_proportions = {k: v / total_nodes for k, v in node_counts.items()}
    node_types = ['S=0 Y=0', 'S=0 Y=1', 'S=1 Y=0', 'S=1 Y=1']

    # 节点数量（整数）
    node_counts_list = [int(node_counts.get((s, y), 0)) for s in [0, 1] for y in [0, 1]]
    # 节点比例（四位小数）
    node_proportions_list = [round(node_proportions.get((s, y), 0), 4) for s in [0, 1] for y in [0, 1]]
    df_nodes = pd.DataFrame([node_counts_list, node_proportions_list], columns=node_types, index=['数量', '比例'])
    print(df_nodes)

    # 统计十种无向边的数量及比例
    edge_counts = Counter()
    for i, j in zip(*adj.nonzero()):
        if i < j:  # 确保每条边只计算一次
            edge_type = (sens[i].item(), labels[i].item(), sens[j].item(), labels[j].item())
            edge_counts[edge_type] += 1

    total_edges = sum(edge_counts.values())
    edge_proportions = {k: v / total_edges for k, v in edge_counts.items()}

    # 定义边的类型
    edge_types = [
        's0y0-s0y0', 's0y1-s0y1', 's1y0-s1y0', 's1y1-s1y1', # s same y same
        's0y0-s1y0', 's0y1-s1y1', # s diff y same
        's0y0-s0y1', 's1y0-s1y1', # s same y diff
        's0y0-s1y1', 's0y1-s1y0'  # s diff y diff
    ]

    # 初始化边计数和比例列表
    edge_counts_list = [0] * 10
    edge_proportions_list = [0] * 10

    # 填充边计数和比例
    for ((s1, y1, s2, y2), count) in edge_counts.items():
        # 确保边的表示是一致的
        edge_type = tuple(sorted([(s1, y1), (s2, y2)]))

        # 构建边的类型字符串
        edge_type_str = f's{int(edge_type[0][0])}y{int(edge_type[0][1])}-s{int(edge_type[1][0])}y{int(edge_type[1][1])}'

        # 获取索引并更新计数
        if edge_type_str in edge_types:
            index = edge_types.index(edge_type_str)
            edge_counts_list[index] += count
            edge_proportions_list[index] += edge_proportions[(s1, y1, s2, y2)]

    # 转换为整数和四位小数
    edge_counts_list = [int(count) for count in edge_counts_list]
    edge_proportions_list = [round(proportion, 4) for proportion in edge_proportions_list]

    # 创建DataFrame
    df_edges = pd.DataFrame([edge_counts_list, edge_proportions_list], columns=edge_types, index=['数量', '比例'])

    # 显示表格
    print('num_edges: ', total_edges)
    print(df_edges)

    # 计算节点度数的分布
    degrees = adj.sum(axis=1).A.flatten()
    degree_distribution = Counter(degrees)
    average_degree = np.mean(degrees)
    print('avg degree: ', average_degree)

    return node_counts, node_proportions, edge_counts, edge_proportions, degree_distribution


def CheckLabel(labels, idx_train, idx_val, idx_test):
    # check the label balance
    print(labels, type(labels))
    value_counts = Counter(labels.numpy())
    value_counts_dict = dict(value_counts)
    print('raw labels:', value_counts_dict)

    labels[labels>1]=1
    value_counts = Counter(labels.numpy())
    value_counts_dict = dict(value_counts)
    print('process labels:', value_counts_dict)

    labels_train = labels[idx_train]
    value_counts = Counter(labels_train.numpy())
    value_counts_dict = dict(value_counts)
    print('train labels:', value_counts_dict)

    labels_val = labels[idx_val]
    value_counts = Counter(labels_val.numpy())
    value_counts_dict = dict(value_counts)
    print('val labels:', value_counts_dict)

    labels_test = labels[idx_test]
    value_counts = Counter(labels_test.numpy())
    value_counts_dict = dict(value_counts)
    print('test labels:', value_counts_dict)


import numpy as np
import scipy.sparse as sp

def add_edges_optimized(adj, sens, labels, num_edges_to_add, edge_type):
    np.random.seed(42)
    
    # 筛选符合条件的节点
    nodes_a = [i for i in range(len(sens)) if sens[i] == edge_type[0] and labels[i] == edge_type[1]]
    nodes_b = [i for i in range(len(sens)) if sens[i] == edge_type[2] and labels[i] == edge_type[3]]

    # 创建潜在边的集合
    potential_edges = set((a, b) for a in nodes_a for b in nodes_b if a < b)

    # 从潜在边中去除已存在的边
    adj_coo = adj.tocoo()
    existing_edges = set(zip(adj_coo.row, adj_coo.col))
    potential_edges -= existing_edges

    # 转换潜在边集合为列表并随机选择要添加的边
    potential_edges_list = list(potential_edges)
    num_edges_to_add = min(num_edges_to_add, len(potential_edges_list))
    selected_indices = np.random.choice(len(potential_edges_list), size=num_edges_to_add, replace=False)
    new_edges = [potential_edges_list[i] for i in selected_indices]

    # 添加边
    rows, cols = zip(*new_edges)
    data = np.ones(len(rows))
    new_edges_matrix = sp.coo_matrix((data, (rows, cols)), shape=adj.shape)

    # 合并新旧矩阵
    adj = adj + new_edges_matrix + new_edges_matrix.T  # 转换为无向边

    return adj.tocsr()


def add_edges(adj, sens, labels, num_edges_to_add, edge_type):
    np.random.seed(42)
    
    adj_lil = adj.tolil()
    possible_edges = []

    for i in range(len(sens)):
        for j in range(i+1, len(sens)):
            if adj_lil[i, j] == 0 and ((sens[i], labels[i], sens[j], labels[j]) == edge_type or 
                                   (sens[j], labels[j], sens[i], labels[i]) == edge_type):
                possible_edges.append((i, j))

    # 检查是否有足够的潜在边可以添加
    num_edges_to_add = min(num_edges_to_add, len(possible_edges))

    if num_edges_to_add > 0:
        new_edges = np.random.choice(range(len(possible_edges)), num_edges_to_add, replace=False)
        for edge_idx in new_edges:
            i, j = possible_edges[edge_idx]
            adj_lil[i, j] = adj_lil[j, i] = 1

    return adj_lil.tocsr()


def remove_edges(adj, sens, labels, num_edges_to_remove, edge_type):
    np.random.seed(42)
    adj_lil = adj.tolil()
    existing_edges = []

    # 找到符合特定类型的已存在边
    for i in range(len(sens)):
        for j in range(i+1, len(sens)):  # 只考虑上三角
            if adj_lil[i, j] != 0 and ((sens[i], labels[i], sens[j], labels[j]) == edge_type or 
                                   (sens[j], labels[j], sens[i], labels[i]) == edge_type):
                existing_edges.append((i, j))

    # 检查是否有足够的边可以移除
    num_edges_to_remove = min(num_edges_to_remove, len(existing_edges))

    if num_edges_to_remove > 0:
        # 随机选择要移除的边
        edges_to_remove = np.random.choice(range(len(existing_edges)), num_edges_to_remove, replace=False)
        for edge_idx in edges_to_remove:
            i, j = existing_edges[edge_idx]
            adj_lil[i, j] = adj_lil[j, i] = 0  # 移除边

    return adj_lil.tocsr()

def remove_edges_efficiently(adj, sens, labels, num_edges_to_remove, edge_type):
    np.random.seed(42)

    # 转换为COO格式进行迭代
    adj_coo = adj.tocoo()
    possible_edges = [(i, j) for i, j in zip(adj_coo.row, adj_coo.col) if
                      ((sens[i], labels[i], sens[j], labels[j]) == edge_type or 
                       (sens[j], labels[j], sens[i], labels[i]) == edge_type) and i < j]

    # 随机选择要移除的边的索引
    num_edges_to_remove = min(num_edges_to_remove, len(possible_edges))
    remove_indices = np.random.choice(len(possible_edges), size=num_edges_to_remove, replace=False)

    # 构建一个标记数组来指示哪些边需要被保留
    keep_mask = np.ones(len(adj_coo.data), dtype=bool)
    for idx in remove_indices:
        i, j = possible_edges[idx]
        keep_mask &= ~((adj_coo.row == i) & (adj_coo.col == j))
        keep_mask &= ~((adj_coo.row == j) & (adj_coo.col == i))  # 由于是无向图

    # 使用布尔索引保留未被移除的边
    adj_coo.data = adj_coo.data[keep_mask]
    adj_coo.row = adj_coo.row[keep_mask]
    adj_coo.col = adj_coo.col[keep_mask]

    # 重新构建稀疏矩阵
    adj_new = sp.coo_matrix((adj_coo.data, (adj_coo.row, adj_coo.col)), shape=adj.shape).tocsr()

    return adj_new