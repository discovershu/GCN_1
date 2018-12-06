from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import traffic_data.synthetic_data_old as synthetic_data
from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 8, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('test_num', 456, 'Number of test nodes.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features0, y_train1, _, _, _, _, test_mask0 = synthetic_data.load_data_hour(week=0, test_num=FLAGS.test_num)
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
    outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
opinion_ = np.zeros_like(y_train1)
opinion = []
opinion_label = []
mean_acc = []
hours = 6
weeks = 35
learning_rate = 0.001
for r in range(10):
    _, features, y_train, _, y_test, train_mask, _, test_mask = synthetic_data.load_data_hour(week=0, test_num=FLAGS.test_num)
    features = sparse_to_tuple(features)
    t1 =time.time()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs):
            p = np.mod(epoch, weeks)
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            #       "train_acc=", "{:.5f}".format(outs[2]))

            # cost_train.append(outs[1])
            # train_acc.append(outs[2])
            # if len(cost_train) == weeks:
            #     cost_epoch_85.append(np.mean(cost_train))
            #     train_acc_85.append(np.mean(train_acc))
            #     cost_train = []
            #     train_acc = []
            #     epoch_85 += 1
            #     print("Iteration:", '%04d' % r, "Epoch:", '%04d' % epoch_85, "train_loss=",
            #           "{:.5f}".format(cost_epoch_85[-1]),
            #           "train_acc=", "{:.5f}".format(train_acc_85[-1]), "time=", "{:.5f}".format(time.time() - t))
            #     if epoch_85 > FLAGS.early_stopping and cost_epoch_85[-1] > np.mean(
            #             cost_epoch_85[-(FLAGS.early_stopping + 1):-1]):
            #         print("Early stopping...")
            #         # break
            # Testing
            # cost_test = []
            # acc_test = []
            # test_prediction = []
            # for i in range(weeks):
            #     # i = i + r * 3
            #     i = np.mod(i, weeks)
            #     _, features, _, _, y_test, _, _, test_mask = read_data.load_data_hour(week=i, test_num=FLAGS.test_num)
            #     # features = preprocess_features(features)
            #     features = sparse_to_tuple(features)
            #     test_cost, test_acc, _, prediction = evaluate(features, support, y_test, test_mask,
            #                                                               placeholders)
            #     cost_test.append(test_cost)
            #     acc_test.append(test_acc)
            #     test_prediction.append(prediction)
            # cost_test = np.mean(cost_test)
            # acc_test = np.mean(acc_test)
            # opinion_p, opinion_l = read_data.get_opinion(test_prediction, test_mask0)
            # opinion_ = opinion_l
            # opinion_label.append(opinion_l)
            # opinion.append(opinion_p)
            # # opinion_error = read_data.opinion_error(opinion_p, test_mask0, weeks=35, hour=hours, p=0)/FLAGS.test_num
            # # print("Iteration:", '%04d' % r, "Test set results:", "cost=", "{:.5f}".format(cost_test),
            # #       "accuracy =", "{:.5f}".format(acc_test * 100), "opinion error=", "{:.5f}".format(opinion_error))
            # # np.save("opinion.npy", opinion)
            # print("Iteration:", '%04d' % r, "Test set results:", "cost=", "{:.5f}".format(cost_test),
            #       "accuracy =", "{:.5f}".format(acc_test * 100))

        # print("GCN Optimization Finished!")

        # Testing
        cost_test = []
        acc_test = []
        cost_train = []
        acc_train = []
        test_prediction = []
        for i in range(1):
            # i = i + r * 3
            i = np.mod(i, weeks)
            # _, features, y_train, _, y_test, train_mask, _, test_mask = synthetic_data.load_data_hour(week=i, test_num=FLAGS.test_num)
            # features = preprocess_features(features)
            # features = sparse_to_tuple(features)
            test_cost, test_acc, test_duration, prediction = evaluate(features, support, y_test, test_mask,
                                                                      placeholders)
            train_cost, train_acc, _, _ = evaluate(features, support, y_train, train_mask, placeholders)
            cost_test.append(test_cost)
            acc_test.append(test_acc)
            cost_train.append(train_cost)
            acc_train.append(train_acc)
            test_prediction.append(prediction)
        cost_test = np.mean(cost_test)
        acc_test = np.mean(acc_test)
        cost_train = np.mean(cost_train)
        acc_train = np.mean(acc_train)
        # opinion_p, opinion_l = read_data.get_opinion(test_prediction, test_mask0)
        # opinion_ = opinion_l
        # opinion_label.append(opinion_l)
        # opinion.append(opinion_p)
        # opinion_error = read_data.opinion_error(opinion_p, test_mask0, weeks=35, hour=hours, p=0)/FLAGS.test_num
        # print("Iteration:", '%04d' % r, "Test set results:", "cost=", "{:.5f}".format(cost_test),
        #       "accuracy =", "{:.5f}".format(acc_test * 100), "opinion error=", "{:.5f}".format(opinion_error))
        # np.save("opinion.npy", opinion)
        print("Iteration:", '%04d' % r, "Test set results:", "cost=", "{:.5f}".format(cost_test),
              "Test accuracy =", "{:.5f}".format(acc_test * 100), "Train accuracy =", "{:.5f}".format(acc_train * 100))
        mean_acc.append(acc_test)
        print(time.time() - t1)
        sess.close()
# cost_test = []
# acc_test = []
# test_prediction = []
# test_label = []
# for r in range(1):
#     for i in range(weeks):
#         _, features, _, _, y_test, _, _, test_mask = read_data.load_data_hour(week=i, test_num=FLAGS.test_num)
#         # features = preprocess_features(features)
#         features = sparse_to_tuple(features)
#         test_cost, test_acc, _, prediction = evaluate(features, support, y_test, test_mask, placeholders)
#         cost_test.append(test_cost)
#         acc_test.append(test_acc)
#         test_prediction.append(prediction)
#         test_label.append(y_test)
# cost_test = np.mean(cost_test)
# acc_test = np.mean(acc_test)
# opinion_p, opinion_l = read_data.get_opinion(test_prediction, test_mask0)
# # opinion_error = read_data.opinion_error(opinion_p, test_mask0, weeks=35, hour=hours, p=0)/FLAGS.test_num
# # print("Final test set results:", "cost=", "{:.5f}".format(cost_test),
# #       "accuracy =", "{:.5f}".format(acc_test * 100), "opinion error=", "{:.5f}".format(opinion_error))
# print("Final test set results:", "cost=", "{:.5f}".format(cost_test),
#       "accuracy =", "{:.5f}".format(acc_test * 100))
print(np.mean(mean_acc))