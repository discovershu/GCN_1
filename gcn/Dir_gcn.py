from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import traffic_data.read_data as read_data
from gcn.utils import *
from gcn.models import GCN, MLP
from collections import Counter
from metrics import masked_acc, masked_dir_error, masked_dir_error2, masked_dir_error3

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' here we don't use this
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 8, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('test_rat', 0.2, 'Number of test nodes.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features0, y_train1, _, _, _, _, test_mask0, _, = read_data.load_data_syn_sub(test_rat=FLAGS.test_rat, index=0)
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


T = 38
print("Model:", FLAGS.model, "test_rat:", FLAGS.test_rat, "T = ", T)
p_list = []
truth = []
acc = []
with tf.Session(config=config) as sess:
    for k in range(T):
        # Train model
        t1 = time.time()
        sess.run(tf.global_variables_initializer())
        # Training part
        # load data
        _, features, y_train, _, y_test, train_mask, _, test_mask, _, = read_data.load_data_syn_sub(
            test_rat=FLAGS.test_rat,
            index=k)
        features = sparse_to_tuple(features)
        for epoch in range(FLAGS.epochs):
            # features = preprocess_features(features)

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            # outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            # print("epoch = ", epoch+1, "training loss:", outs[1], "training accuracy:", outs[2])
            # if np.mod(epoch+1, 100) == 0:
            #     outs = evaluate(features, support, y_test, test_mask, placeholders)
            #     print("epoch = ", epoch + 1, "test loss:", outs[0], "test accuracy:", outs[1])
        outs = evaluate(features, support, y_test, test_mask, placeholders)
        p_list.append(outs[3])
        truth.append(y_test)
        acc.append(outs[1])
        print("index = ", k + 1, "test loss:", outs[0], "test accuracy:", outs[1])
    error2 = masked_dir_error2(p_list, truth, test_mask0)
    error = masked_dir_error(p_list, truth, test_mask0)
    error3 = masked_dir_error3(p_list, truth, test_mask0)


    print("Dir error:", "{:.2f}".format(error), "Dir error2:", "{:.3f}".format(error2), "Dir error3:", "{:.3f}".format(error3), "accurate:", "{:.2f}".format(np.mean(acc)))
