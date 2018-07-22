import tensorflow as tf
import tensorflow.contrib.slim as slim


# noinspection SpellCheckingInspection
class QNetwork(object):

    def __init__(self, n_hidden, rnn_cell, scope, learning_rate=0.0001):
        # The network receives a frame from the game, flattened into an array.
        # It then resizes and processes it through four convolutional layers.
        self.scalar_input = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.image_in = tf.reshape(self.scalar_input, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.image_in, num_outputs=32,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                                 biases_initializer=None, scope=scope + '_conv1')
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64,
                                 kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                 biases_initializer=None, scope=scope + '_conv2')
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 biases_initializer=None, scope=scope + '_conv3')
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=n_hidden,
                                 kernel_size=[7, 7], stride=[1, 1], padding='VALID',
                                 biases_initializer=None, scope=scope + '_conv4')

        self.sequence_length = tf.placeholder(dtype=tf.int32)

        # Take the output from the final convolutional layer and send it
        # to a recurrent layer. The input must be reshaped into
        # [batch, trace, units] for RNN processing, and then returned to
        # [batch, units] when sent through the upper levels.
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.conv_flat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.sequence_length, n_hidden])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.conv_flat, cell=rnn_cell, dtype=tf.float32,
                                                     initial_state=self.state_in, scope=scope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, n_hidden])

        # The output from the recurrent layer is then split into separate
        # Advantage and Value streams.
        self.stream_a, self.stream_v = tf.split(self.rnn, 2, 1)
        self.aw = tf.Variable(tf.random_normal([n_hidden // 2, 4]))
        self.vw = tf.Variable(tf.random_normal([n_hidden // 2, 1]))
        self.advantage = tf.matmul(self.stream_a, self.aw)
        self.value = tf.matmul(self.stream_v, self.vw)
        self.salience = tf.gradients(self.advantage, self.image_in)

        # then combine them together to get our final Q-values
        self.q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.q_out, 1)

        # obtain the loss by taking the sum of squares difference between
        # the target and predicted Q-values
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.target_q - self.q)

        # In order to only propagate accurate gradients through the network,
        # we will mask the first half of the losses for each trace as per
        # Lample & Chatlot 2016
        self.mask_a = tf.zeros([self.batch_size, self.sequence_length // 2])
        self.mask_b = tf.ones([self.batch_size, self.sequence_length // 2])
        self.mask = tf.concat([self.mask_a, self.mask_b], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update_op = self.optimizer.minimize(self.loss)
