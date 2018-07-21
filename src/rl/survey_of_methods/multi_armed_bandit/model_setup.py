import tensorflow as tf


class Agent(object):
    """
    A simple neural agent. It consists of a set of values for each of the bandits.
    Each value is an estimate of the value of the return from choosing the bandit.
    We use a policy gradient method to update the agent by moving the value for
    the selected action toward the received reward.
    """

    def __init__(self, n_bandits, learning_rate=0.001):
        # Establish the feed-forward part of the network. This does the actual choosing.
        self.w = tf.Variable(tf.ones([n_bandits]))
        self.a = tf.argmax(self.w, 0)
        self.reward_ph = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_ph = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.w, self.action_ph, [1])
        loss = -tf.log(self.responsible_weight) * self.reward_ph
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update_op = optimizer.minimize(loss)
