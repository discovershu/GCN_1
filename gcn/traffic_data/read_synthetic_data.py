import numpy as np
import random
from scipy import sparse
from collections import Counter
import synthetic_supervised_data as syn_data


def get_hotvalue(label):
    hotvalue = np.zeros([len(label), 2])
    for i in range(len(label)):
        if label[i] == 1.0:  # conjestion
            # feat[i] = 100.0
            hotvalue[i] = [0, 1]
        elif label[i] == 0.0:  # non-conjestion
            hotvalue[i] = [1, 0]
    return hotvalue


def load_train_data_synthetic(k, label_all, features_all):
    random.seed(123)
    np.random.seed(123)
    adj_n = np.load(
        "/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_adjacency_matrix.npy")
    feature = features_all[k]
    label = label_all[k]
    train_label = get_hotvalue(label)

    train_mask = np.ones_like(feature, dtype=bool)

    adj = sparse.csr_matrix(adj_n)
    features = sparse.csr_matrix(np.reshape(feature, [len(feature), 1]))

    return adj, features, train_label, train_mask


def load_test_data_synthetic(k, label_all, features_all):
    random.seed(123)
    np.random.seed(123)
    adj_n = np.load(
        "/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_adjacency_matrix.npy")
    feature = features_all[k]
    label = label_all[k]
    test_label = get_hotvalue(label)

    test_mask = np.zeros_like(feature, dtype=bool)
    for i in range(len(label)):
        if label[i] == 1.0:
            test_mask[i] = True
    adj = sparse.csr_matrix(adj_n)
    features = sparse.csr_matrix(np.reshape(feature, [len(feature), 1]))

    return adj, features, test_label, test_mask


def load_train_data_synthetic1(k):
    adj_n = np.load(
        "/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_adjacency_matrix.npy")
    features_all = np.load(
        "/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_train_feature.npy")
    label_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_train_label.npy")
    feature = features_all[k]
    label = label_all[k]
    train_label = get_hotvalue(label)

    train_mask = np.ones_like(feature, dtype=bool)

    adj = sparse.csr_matrix(adj_n)
    features = sparse.csr_matrix(np.reshape(feature, [len(feature), 1]))

    return adj, features, train_label, train_mask


def load_test_data_synthetic1(k):
    adj_n = np.load(
        "/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_adjacency_matrix.npy")
    features_all = np.load(
        "/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_test_feature.npy")
    label_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_copy1/gcn/traffic_data/synthetic_test_label.npy")
    feature = features_all[k]
    label = label_all[k]
    test_label = get_hotvalue(label)

    test_mask = np.ones_like(feature, dtype=bool)

    adj = sparse.csr_matrix(adj_n)
    features = sparse.csr_matrix(np.reshape(feature, [len(feature), 1]))

    return adj, features, test_label, test_mask


def get_F1_score(pred, mask):
    mask = np.asarray(mask, dtype=float)
    pred = np.round(pred)
    pred = pred[:, 1]
    TP = np.sum(pred * mask)
    FP = np.sum(pred) - TP
    FN = np.sum(mask) - TP
    TN = len(mask) - TP - FP - FN
    if (TP + FP) == 0.0:
        precision = 1.0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if (precision + recall) == 0.0:
        F1_score = 0.0
    else:
        F1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, F1_score

if __name__ == '__main__':
    aa = np.ones([100, 2])
    a = aa[:, 1]
    print(1)