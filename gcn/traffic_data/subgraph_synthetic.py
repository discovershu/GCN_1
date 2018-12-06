import numpy as np
import random
from scipy import sparse
from collections import Counter


def generate_adjacency_matrix():
    graph_size = 10
    adj_size = graph_size * graph_size
    adj = np.zeros([adj_size, adj_size])
    for i in range(len(adj)):
        left = i - 1
        right = i + 1
        up = i - graph_size
        down = i + graph_size
        p = i / graph_size
        left_t = p * graph_size
        right_t = (p + 1) * graph_size
        if left_t <= left < right_t:
            adj[i][left] = 1
        if left_t <= right < right_t:
            adj[i][right] = 1
        if 0 <= up < adj_size:
            adj[i][up] = 1
        if 0 <= down < adj_size:
            adj[i][down] = 1
    return adj


def get_neigh(adj):
    neigh = []
    for i in range(len(adj)):
        neigh_i = []
        for j in range(len(adj[i])):
            jj = adj[i][j]
            if jj == 1:
                neigh_i.append(j)
        neigh.append(neigh_i)
    return neigh


def get_subgraph(num):
    adj = np.load("syn_adj_100.npy")
    neigh = np.load("syn_neigh_100.npy")
    random_point = int(random.randrange(len(adj)))
    sub_1 = []
    sub_2 = []
    k = 0
    while k < num:
        neigh_i = neigh[random_point]
        random_next = random.sample(neigh_i, 1)
        if random_next[0] in sub_1:
            pass
        else:
            sub_1.append(random_next[0])
            k = k + 1
        random_point = int(random_next[0])

    random_point = int(random.randrange(len(adj)))
    k = 0
    while k < num:
        neigh_i = neigh[random_point]
        random_next = random.sample(neigh_i, 1)
        if random_next[0] in sub_1:
            pass
        elif random_next[0] in sub_2:
            pass
        else:
            sub_2.append(random_next[0])
            k = k + 1
        random_point = int(random_next[0])

    return sub_1, sub_2


def category_dis(p1, p2, p3):
    label = []
    random_point = int(random.randrange(1000))
    if random_point < p1 * 1000:
        label = [1, 0, 0]
    elif random_point < ((p1 + p2) * 1000):
        label = [0, 1, 0]
    else:
        label = [0, 0, 1]
    return label


def get_obs(T):
    adj = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/syn_adj_100.npy")
    sub_1 = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/subgraph_1.npy")
    sub_2 = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/subgraph_2.npy")
    label_T = []
    for k in range(T):
        label = []
        for i in range(len(adj)):
            if i in sub_1:
                label.append(category_dis(0.7, 0.2, 0.1))
            elif i in sub_2:
                label.append(category_dis(0.1, 0.7, 0.2))
            else:
                label.append(category_dis(0.2, 0.1, 0.7))
        label_T.append(label)
    return label_T


def get_obs_dif(T):
    adj = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/syn_adj_100.npy")
    sub_1 = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/subgraph_1.npy")
    sub_2 = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/subgraph_2.npy")
    label_T = []
    for i in range(len(adj)):
        t = int(random.randrange(T+1))
        label = []
        if t>0:
            if i in sub_1:
                for j in range(t):
                    label.append(category_dis(0.7, 0.2, 0.1))
            elif i in sub_2:
                for j in range(t):
                    label.append(category_dis(0.1, 0.7, 0.2))
            else:
                for j in range(t):
                    label.append(category_dis(0.2, 0.1, 0.7))
            label = np.asarray(label)
            label = np.sum(label, axis=0) + 1
        else:
            label = [1, 1, 1]
        label_T.append(label)
    label_T = np.asarray(label_T)
    return label_T

if __name__ == '__main__':
    # a = generate_adjacency_matrix()
    # np.save("syn_adj_100.npy", a)
    # adj = np.load("syn_adj_100.npy")
    # n = get_neigh(adj)
    # np.save("syn_neigh_100.npy", n)
    # s1, s2 = get_subgraph(20)
    # np.save("subgraph_1.npy", s1)
    # np.save("subgraph_2.npy", s2)
    # T = get_obs(38)
    # np.save("label_38", T)
    # generate_adjacency_matrix()
    b = get_obs_dif(38)
    np.save("/network/rit/lab/ceashpc/xujiang/project/Dir_synthitic/label_38_dif.npy", b)
    print 1
