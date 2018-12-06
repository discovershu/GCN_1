from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import traffic_data.read_synthetic_data as read_data
import traffic_data.synthetic_supervised_data as syn_data

from gcn.utils import *
from gcn.models import GCN, MLP
from collections import Counter
import random

# Set random seed
seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' here we don't use this
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 8, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 7, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('test_rat', 0.2, 'Number of test nodes.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# print(adj[106])
# print(adj[2461])
# label_train, features_train, label_test, features_test = syn_data.generate_train_test_data()
adj, label_train, features_train, label_test, features_test = syn_data.generate_train_test_data_noise(0.1)
_, features0, y_train1, test_mask0 = read_data.load_train_data_synthetic(0, label_train, features_train)
# Some preprocessing
# features0 = preprocess_features(features0)
features0 = sparse_to_tuple(features0)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features0[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train1.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features0[2][1], logging=True)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pppp], feed_dict=feed_dict_val)
    # outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


weeks = 600  # T = weeks
window_slide = 1  # window size
opinion_e = []
acc = []
with tf.Session(config=config) as sess:
    for j in range(1):

        cost_val = []
        cost_train = []
        cost_epoch_85 = []
        train_acc = []
        train_acc_85 = []
        epoch_85 = 0
        F1 = []
        for k in range(10):
            # Train model
            random.seed(k)
            np.random.seed(k)
            tf.set_random_seed(k)
            adj, label_train, features_train, label_test, features_test = syn_data.generate_train_test_data_noise(0.1)
            t1 = time.time()
            sess.run(tf.global_variables_initializer())
            # Training part
            for epoch in range(FLAGS.epochs):
                # more example
                # p = np.mod(epoch, weeks) + j
                p = np.mod(epoch, weeks)  # fix the uncertain T
                # load data
                _, features, y_train, train_mask = read_data.load_train_data_synthetic(p, label_train, features_train)
                # features = preprocess_features(features)
                features = sparse_to_tuple(features)
                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                # Training step
                # outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
                outs = sess.run([model.opt_op, model.loss, model.accuracy, model.pppp], feed_dict=feed_dict)

                # print("epoch = ", epoch , "Train Accuary=", "{:.5f}".format(outs[2]), "Train Loss =", "{:.5f}".format(outs[1]))
                cost_train.append(outs[1])
                train_acc.append(outs[2])
                if len(cost_train) == 100:
                    cost_epoch_85.append(np.mean(cost_train))
                    train_acc_85.append(np.mean(train_acc))
                    cost_train = []
                    train_acc = []
                    epoch_85 += 1
                    print("Epoch:", '%04d' % epoch_85, "train_loss=",
                          "{:.5f}".format(cost_epoch_85[-1]),
                          "train_acc=", "{:.5f}".format(train_acc_85[-1]))
            # print("Optimization Finished!")

            # Testing part
            cost_test = []
            acc_test = []
            test_prediction = []
            y_truth = []
            precision = []
            recall = []
            F1_score = []
            for i in range(100):
                # i = i + k
                _, features, y_test, test_mask = read_data.load_test_data_synthetic(i, label_test, features_test)
                # features = preprocess_features(features)
                features = sparse_to_tuple(features)
                test_cost, test_acc, test_duration, prediction = evaluate(features, support, y_test, test_mask,
                                                                          placeholders)
                cost_test.append(test_cost)
                acc_test.append(test_acc)
                test_prediction.append(prediction)
                y_truth.append(y_test)
                p, r, f1 = read_data.get_F1_score(prediction, test_mask)
                precision.append(p)
                recall.append(r)
                F1_score.append(f1)
            cost_test = np.mean(cost_test)
            acc_test = np.mean(acc_test)
            F1.append(np.mean(F1_score))
            # opinion_gcn = read_data.opinion_c(test_prediction)
            # opinion_truth = read_data.opinion_c(y_truth)
            # error_opinion = read_data.opinion_error(opinion_gcn, opinion_truth, test_mask0)
            # print("Hours:", j, "Test set results:", "cost=", "{:.5f}".format(cost_test),
            #       "accuracy=", "{:.5f}".format(acc_test * 100), "opinion error=", "{:.5f}".format(error_opinion))
            # opinion_e.append(error_opinion)
            # acc.append(acc_test)
            # np.save("./result/opinion_0.3_dc_T11_gcn_6hour_Fr.npy", opinion_e)
            t2 = time.time()
            # print("running time = ", t2 - t1)
            # print("weeks:", '%02d' % weeks, "window:", '%02d' % k, "Mean opinion error=", "{:.5f}".format(np.mean(opinion_e)))
            print("Test precision=", "{:.5f}".format(np.mean(precision)), "Test Recall=", "{:.5f}".format(np.mean(recall)),
                  "Test F1_score=", "{:.5f}".format(np.mean(F1_score)))
        print("10 times Average F1_score=", "{:.5f}".format(np.mean(F1)))
# print("weeks:", '%02d' % weeks, "test num:", '%03d' % FLAGS.test_num, "Mean opinion error=", "{:.5f}".format(np.mean(opinion_e)))
# print("weeks:", '%02d' % weeks, "test num:", '%03d' % FLAGS.test_num, "Mean Accuary=", "{:.5f}".format(np.mean(acc)))
