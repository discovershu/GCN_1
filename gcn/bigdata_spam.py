from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import traffic_data.read_spam as read_data
from gcn.utils import *
from gcn.models import GCN, MLP, GCN2
from collections import Counter
from scipy import sparse

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' here we don't use this
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 1, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 8, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('test_ratio', 0.1, 'test number of dataset.')
flags.DEFINE_float('noise', 0.0, 'noise of synthetic data.')
flags.DEFINE_integer('bias', 0, 'bias.')
flags.DEFINE_integer('node', 200000, 'node size.')


# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# adj, features0, y_train1, _, _, _, _, test_mask0, _, _, _ = read_data.generate_train_test_spam_noise_sample
adj, label_b, label_u, train_mask0, test_mask0 = read_data.generate_train_test_spam_noise_sample(FLAGS.test_ratio, FLAGS.node, FLAGS.noise)

print("test_ratio:", FLAGS.test_ratio, "noise:", FLAGS.noise, "node:", FLAGS.node)

# Some preprocessing
# features0 = preprocess_features(features0)
features0 = label_b[0] * train_mask0[0]
features0 = np.reshape(features0, [len(features0), 1])
features0 = sparse.csr_matrix(features0)
features0 = sparse_to_tuple(features0)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN2
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
    'labels': tf.placeholder(tf.float32, shape=(None, 1)),
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

cost_all =[]

with tf.Session(config=config) as sess:
    for j in range(1):
        for k in range(6):
            # Train model
            sess.run(tf.global_variables_initializer())
            features = label_b[k] * train_mask0[k]
            features = np.reshape(features, [len(features), 1])
            features = sparse.csr_matrix(features)
            features = sparse_to_tuple(features)
            y_train = label_b[k]
            y_train = np.reshape(y_train, [-1, 1])
            train_mask = train_mask0[k]
            test_mask = test_mask0[k]
            # Training part
            for epoch in range(FLAGS.epochs):
                feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                outs = sess.run([model.opt_op, model.cost], feed_dict=feed_dict)
                print(outs[1])
            print("Optimization Finished!")

            # Testing part
            feed_dict = construct_feed_dict(features, support, y_train, test_mask, placeholders)
            outs = sess.run([model.cost], feed_dict=feed_dict)
            cost_all.append(outs[0])
        print("error=", np.mean(cost_all))
