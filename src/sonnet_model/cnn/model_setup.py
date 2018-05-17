import sonnet as snt
import tensorflow as tf


class MyCNN(snt.AbstractModule):

    def __init__(self, name='my_cnn'):
        super().__init__(name=name)

    def _build(self, x):
        w1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        w2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

        z1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
        a1 = tf.nn.relu(z1)
        p1 = tf.nn.max_pool(a1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
        z2 = tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding='SAME')
        a2 = tf.nn.relu(z2)
        p2 = tf.nn.max_pool(a2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        p = tf.contrib.layers.flatten(p2)
        z3 = tf.contrib.layers.fully_connected(p, 6, activation_fn=None)
        parameters = {'w1': w1, 'w2': w2}
        return z3, parameters


def compute_cost(z3, y_placeholder):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y_placeholder))


def model_builder(x_placeholder, y_placeholder, learning_rate=0.009):
    model = MyCNN()
    z3, _ = model(x_placeholder)
    cost_op = compute_cost(z3, y_placeholder)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)
    return cost_op, optimizer, z3
