from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import traffic_data.read_data as read_data
from gcn.utils import *
from gcn.models import GCN, MLP
from collections import Counter

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' here we don't use this
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 8, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('test_rat', 0.8, 'Number of test nodes.')


# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features0, y_train1, _, _, _, _, test_mask0, _, _, _ = read_data.load_data_pa(week=0, test_rat=FLAGS.test_rat, hour=0, k=0)
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
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


weeks = 38  # T = weeks
window_slide = 43 - weeks +1  # window size
opinion_e = []
acc = []
belief_error = []
uncertain_error = []
print("0:", "test_rat:", FLAGS.test_rat)
with tf.Session(config=config) as sess:
    for j in range(6):

        cost_val = []
        cost_train = []
        cost_epoch_85 = []
        train_acc = []
        train_acc_85 = []
        epoch_85 = 0
        for k in range(window_slide):
            # Train model
            t1 = time.time()
            sess.run(tf.global_variables_initializer())
            # Training part
            for epoch in range(FLAGS.epochs):
                # more example
                # p = np.mod(epoch, weeks) + j
                p = np.mod(epoch, weeks) + k  # fix the uncertain T
                # load data
                _, features, y_train, _, _, train_mask, _, _, _, _, _ = read_data.load_data_pa(week=p, test_rat=FLAGS.test_rat,
                                                                                        hour=j, k=k)
                # features = preprocess_features(features)
                features = sparse_to_tuple(features)
                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                # Training step
                # outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
                outs = sess.run([model.opt_op, model.loss, model.accuracy, model.pppp], feed_dict=feed_dict)
                # print(outs[2])
            print("Optimization Finished!")

            # Testing part
            cost_test = []
            acc_test = []
            test_prediction = []
            y_truth = []
            for i in range(weeks):
                i = i + k
                _, features, _, _, y_test, _, _, test_mask, test_index, belief, uncertain = read_data.load_data_pa(week=i, test_rat=FLAGS.test_rat,
                                                                                      hour=j, k=k)
                # features = preprocess_features(features)
                features = sparse_to_tuple(features)
                test_cost, test_acc, test_duration, prediction = evaluate(features, support, y_test, test_mask,
                                                                          placeholders)
                cost_test.append(test_cost)
                acc_test.append(test_acc)
                test_prediction.append(prediction)
                y_truth.append(y_test)
            cost_test = np.mean(cost_test)
            acc_test = np.mean(acc_test)
            opinion_gcn = read_data.opinion_dcpa(test_prediction)
            b_error, u_error = read_data.get_error_dcpa(opinion_gcn, belief, uncertain, test_index, weeks)
            # opinion_truth = read_data.opinion_c(y_truth)
            # error_opinion = read_data.opinion_error(opinion_gcn, opinion_truth, test_mask0)
            # print("Hours:", j, "Test set results:", "cost=", "{:.5f}".format(cost_test),
            #       "accuracy=", "{:.5f}".format(acc_test * 100), "opinion error=", "{:.5f}".format(error_opinion))
            # opinion_e.append(error_opinion)
            # acc.append(acc_test)
            # np.save("./result/opinion_0.3_dc_T11_gcn_6hour_Fr.npy", opinion_e)
            print("belief error=", b_error, "uncertain error=", u_error)
            belief_error.append(b_error)
            uncertain_error.append(u_error)
            t2 = time.time()
            print(t2 - t1)
            # print("weeks:", '%02d' % weeks, "window:", '%02d' % k, "Mean opinion error=", "{:.5f}".format(np.mean(opinion_e)))
            # print("weeks:", '%02d' % weeks, "window:", '%02d' % k, "Mean Accuary=",
            #       "{:.5f}".format(np.mean(acc)))
        print("5:", '%02d' % weeks, "test_rat:", FLAGS.test_rat, "belief error=", np.mean(belief_error), "uncertain error=", np.mean(uncertain_error))
