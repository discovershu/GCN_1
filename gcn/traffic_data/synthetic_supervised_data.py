import numpy as np
import random
from scipy import sparse
from collections import Counter
import networkx as nx


def generate_adjacency_matrix():
    graph_size = 10
    adj_size = graph_size * graph_size
    adj = np.zeros([adj_size, adj_size])
    for i in range(len(adj)):
        left = i - 1
        right = i + 1
        up = i - graph_size
        down = i + graph_size
        p = i/graph_size
        left_t = p * graph_size
        right_t = (p+1) * graph_size
        if left_t <= left < right_t:
            adj[i][left] = 1
        if left_t <= right < right_t:
            adj[i][right] = 1
        if 0 <= up < adj_size:
            adj[i][up] = 1
        if 0 <= down < adj_size:
            adj[i][down] = 1
    return adj


# def generate_feature():
#     adj = generate_adjacency_matrix()
#     label = np.zeros(len(adj))
#     feature = np.zeros(len(adj))
#     rand_index = random.sample(range(len(label)), 1)
#     subgraph = [x for x in rand_index]
#     # subgraph.append(rand_index[0])
#     for i in range(len(label)):
#         for j in rand_index:
#             if adj[j][i] == 1:
#                 subgraph.append(i)
#
#     for n in range(len(label)):
#         if n in subgraph:
#             label[n] = 1.0
#             feature[n] = np.random.normal(2, 1)
#         else:
#             label[n] = 0.0
#             feature[n] = np.random.normal(0, 1)
#     return label, feature


def generate_subgraph(graph, center, step_size):
    subgraph = []
    subgraph.append(center)
    # cand_queue = deque()
    # cand_queue.append(center)
    for i in range(step_size):
        neighbors = list(set([node for cur_node in subgraph for node in nx.neighbors(graph, cur_node) if node not in subgraph]))
        # print(neighbors)
        subgraph.extend(neighbors)

        # size = len(cand_queue)
        # if size > 0:
        #     for j in range(size):
        #         cur_node = cand_queue.popleft()
        #         # print('current node: ', cur_node)
        #         neighbors = [node for node in nx.neighbors(graph, cur_node)]
        #         subgraph.extend(neighbors)
        #         # print('updated subgraph: ', subgraph)
        #         cand_queue.extend(neighbors)
        #         # print('updated candidate queue: ', cand_queue)

    # print(subgraph)
    return list(set(subgraph))


# def generate_feature_noise(noise):
#     adj = generate_adjacency_matrix()
#     label = np.zeros(len(adj))
#     feature = np.zeros(len(adj))
#     rand_index = random.sample(range(len(label)), 4)
#     noise_index = random.sample(range(len(label)), int(noise * len(label)))
#     subgraph = [x for x in rand_index]
#     for i in range(len(label)):
#         for j in rand_index:
#             if adj[j][i] == 1:
#                 subgraph.append(i)
#     mu_0 = 0
#     mu_1 = 2.0
#     for n in range(len(label)):
#         if n in subgraph:
#             label[n] = 1.0
#             feature[n] = np.random.normal(mu_1, 1)
#         else:
#             label[n] = 0.0
#             feature[n] = np.random.normal(mu_0, 1)
#     for m in noise_index:
#         if m in subgraph:
#             feature[m] = np.random.normal(mu_0, 1)
#         else:
#             feature[m] = np.random.normal(mu_1, 1)
#     return label, feature


def generate_feature_noise_stepsize(noise):
    adj = generate_adjacency_matrix()
    adj_nx = nx.from_numpy_matrix(adj)
    label = np.zeros(len(adj))
    feature = np.zeros(len(adj))
    rand_index = random.sample(range(len(label)), 1)
    noise_index = random.sample(range(len(label)), int(noise * len(label)))
    subgraph = generate_subgraph(adj_nx, rand_index[0], 4)
    # subgraph = [x for x in rand_index]
    # for i in range(len(label)):
    #     for j in rand_index:
    #         if adj[j][i] == 1:
    #             subgraph.append(i)
    mu_0 = 0
    mu_1 = 2
    for n in range(len(label)):
        if n in subgraph:
            label[n] = 1.0
            feature[n] = np.random.normal(mu_1, 1)
        else:
            label[n] = 0.0
            feature[n] = np.random.normal(mu_0, 1)
    for m in noise_index:
        if m in subgraph:
            feature[m] = np.random.normal(mu_0, 1)
        else:
            feature[m] = np.random.normal(mu_1, 1)
    return label, feature



def generate_train_test_data_noise(noise):
    train_label = []
    train_feature = []
    test_label = []
    test_feature = []
    for i in range(1000):
        label, feature = generate_feature_noise_stepsize(noise)
        train_label.append(label)
        train_feature.append(feature)
    for j in range(100):
        label, feature = generate_feature_noise_stepsize(noise)
        test_label.append(label)
        test_feature.append(feature)
    adj_n = generate_adjacency_matrix()
    adj = sparse.csr_matrix(adj_n)
    return adj, train_label, train_feature, test_label, test_feature


if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    generate_feature_noise_stepsize(0.1)
    # train_label, train_feature, test_label, test_feature = generate_train_test_data()
    # np.save("synthetic_train_label", train_label)
    # np.save("synthetic_train_feature", train_feature)
    # np.save("synthetic_test_label", test_label)
    # np.save("synthetic_test_feature", test_feature)
    # adj = generate_adjacency_matrix()
    # np.save("synthetic_adjacency_matrix", adj)
    print(1)
