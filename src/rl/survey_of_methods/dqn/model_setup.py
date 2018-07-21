import tensorflow as tf
import tensorflow.contrib.slim as slim


class QNetwork(object):

    def __init__(self, n_hidden, n_actions, learning_rate=0.0001):
        # The network receives a frame from the game, flattened into an array.
        # It then resizes and processes it through four convolutional layers.
        self.scalar_input = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.image_in = tf.reshape(self.scalar_input, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.image_in, num_outputs=32,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64,
                                 kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=n_hidden,
                                 kernel_size=[7, 7], stride=[1, 1], padding='VALID',
                                 biases_initializer=None)

        # take the output from the final convolutional layer and split it into
        # separate advantage and value streams
        self.stream_ac, self.stream_vc = tf.split(self.conv4, 2, 3)
        self.stream_a = slim.flatten(self.stream_ac)
        self.stream_v = slim.flatten(self.stream_vc)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.aw = tf.Variable(xavier_init([n_hidden // 2, n_actions]))
        self.vw = tf.Variable(xavier_init([n_hidden // 2, 1]))
        self.advantage = tf.matmul(self.stream_a, self.aw)
        self.value = tf.matmul(self.stream_v, self.vw)

        # then combine them together to get our final Q-values
        self.q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.q_out, 1)

        # obtain the loss by taking the sum of squares difference between
        # the target and predicted Q-values
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32)

        self.q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.target_q - self.q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_op = self.trainer.minimize(self.loss)
