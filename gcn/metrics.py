import tensorflow as tf
import numpy as np


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_acc(preds, labels, mask):
    preds = np.argmax(preds, 1)
    labels = np.argmax(labels, 1)
    cor = np.equal(preds, labels)
    cor = np.asarray(cor, dtype=float)
    mask = np.asarray(mask, dtype=float)
    mask = mask / np.mean(mask)
    acc = cor * mask
    acc = np.mean(acc)
    return acc


def masked_mse_square(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    loss = tf.square(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy2(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_mae_abs(preds, labels, mask):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = tf.abs(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_decode(preds, mask):
    """MSE decode with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    loss = mask - preds
    mask /= tf.reduce_mean(mask)
    loss = loss * mask
    return tf.reduce_mean(loss)


def masked_decode_sparse(pred, adj):
    logits = tf.convert_to_tensor(pred, name="logits")
    targets = tf.convert_to_tensor(adj, name="targets")
    # targets = tf.reshape(targets, [70317, 70317])
    try:
        targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError(
            "logits and targets must have the same shape (%s vs %s)" %
            (logits.get_shape(), targets.get_shape()))
    loss = targets - logits
    targets /= tf.reduce_mean(targets)
    loss = loss * targets
    return tf.reduce_mean(loss)


def masked_mae_rnn(preds, labels, mask):
    loss = tf.abs(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_mse_rnn(preds, labels, mask):
    loss = tf.square(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_dir_error(pred, labels, mask):
    prod = np.prod(pred, axis=0)
    prod_pow = np.power(prod, 1.0 / len(pred)) * (len(pred) + 3)
    label = np.sum(labels, axis=0) + 1
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    loss = np.square(prod_pow - label)
    loss = np.mean(loss, axis=1)
    loss *= mask
    return np.mean(loss)


def masked_dir_error2(pred, labels, mask):
    prod = np.prod(pred, axis=0)
    prod_pow = np.power(prod, 1.0 / len(pred))
    label = (np.sum(labels, axis=0) + 1) / float(len(labels) + 3)
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    loss = np.square(prod_pow - label)
    loss = np.mean(loss, axis=1)
    loss *= mask
    return np.mean(loss)


def masked_dir_error3(pred, labels, mask):
    S = len(pred) + 3.0
    prod = np.prod(pred, axis=0)
    prod_pow = np.power(prod, 1.0 / len(pred))
    prod_pow = prod_pow - 1.0/S
    label = (np.sum(labels, axis=0)) / float(len(labels) + 3.0)
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    loss = np.abs(prod_pow - label)
    loss = np.mean(loss, axis=1)
    loss *= mask
    return np.mean(loss)


def get_label(pred):
    arg = np.argmax(pred, axis=2)
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if arg[i][j] == 0:
                pred[i][j] = [1, 0, 0]
            elif arg[i][j] == 1:
                pred[i][j] = [0, 1, 0]
            else:
                pred[i][j] = [0, 0, 1]
    return pred


def masked_dir_error_semi(pred, labels, mask):
    pred = get_label(pred)
    evidence_l = np.sum(labels, axis=0)
    S_l = np.sum(evidence_l, axis=1) + 3.0
    evidence_p = np.sum(pred, axis=0)
    S_p = np.sum(evidence_p, axis=1) + 3.0
    b1_l = evidence_l[:, 0] / S_l
    b2_l = evidence_l[:, 1] / S_l
    b3_l = evidence_l[:, 2] / S_l
    u_l = 3 / S_l
    b1_p = evidence_p[:, 0] / S_p
    b2_p = evidence_p[:, 1] / S_p
    b3_p = evidence_p[:, 2] / S_p
    u_p = 3 / S_p
    loss = np.square(b1_l - b1_p) + np.square(b2_l - b2_p) + np.square(b3_l - b3_p) + np.square(u_l - u_p)
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    loss *= mask

    return np.mean(loss)
