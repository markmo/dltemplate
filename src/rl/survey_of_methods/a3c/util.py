import numpy as np
import scipy.misc
import scipy.signal
import tensorflow as tf


def discount(x, gamma):
    """ Calculate discounted returns """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.):
    """ Initialize weights for policy and value output layers """

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def process_frame(frame):
    """ Preprocess Doom screen image to produce cropped and resized image """
    s = frame[10:-10, 30:-30]
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [np.prod(s.shape)]) / 255.
    return s


def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another. Used to set worker network parameters
    to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder

