import tensorflow as tf


def get_inputs(constants):
    input_x = tf.placeholder(tf.float32, shape=[None, constants['n_input']], name='input_X')
    input_y = tf.placeholder(tf.float32, shape=[None, constants['n_classes']], name='input_y')
    return input_x, input_y
