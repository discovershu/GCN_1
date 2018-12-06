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


def generate_synthetic_data(num, nosie):
    random.seed(123)
    adjacency_l = get_adjacency_list()
    syn_feature = -np.ones(len(adjacency_l))
    random_point = int(random.randrange(1522))
    random_data = []
    k = 0
    while k < num:
        neigh = adjacency_l[random_point]
        random_next = random.sample(neigh, 1)
        if random_next[0] in random_data:
            pass
        else:
            random_data.append(random_next[0])
            k = k + 1
            syn_feature[int(random_next[0])] = 1.0
        random_point = int(random_next[0])
    nosie_feat = np.array(syn_feature)
    noise_index = random.sample(range(len(adjacency_l)), int(nosie * len(adjacency_l)))
    for item in noise_index:
        item = int(item)
        nosie_feat[item] = -syn_feature[item]
    return syn_feature, nosie_feat


def get_neigh(adj):
    neigh = []
    for i in range(len(adj)):
        neigh_i = []
        for j in adj[i]:
            if j == 1:
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


if __name__ == '__main__':
    # a = generate_adjacency_matrix()
    # np.save("syn_adj_100.npy", a)
    # adj = np.load("syn_adj_100.npy")
    # n = get_neigh(adj)
    # np.save("syn_neigh_100.npy", n)

    print 1
