import tensorflow as tf
import tensorflow.contrib.slim as slim


class Agent(object):

    def __init__(self, learning_rate, state_dim, n_actions, n_hidden):
        self.state_in = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, n_hidden, activation_fn=tf.nn.relu,
                                      biases_initializer=None)
        self.output = slim.fully_connected(hidden, n_actions, activation_fn=tf.nn.softmax,
                                           biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        self.reward_ph = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_ph = tf.placeholder(shape=[None], dtype=tf.int32)

        # output shape is [num episodes until done, n_actions]
        # select the output value per row corresponding to the
        # index of the chosen action, similar to numpy operation
        # A[np.arange(A.shape[0]), indices]
        row_indices = tf.range(tf.shape(self.action_ph)[0])
        indices = tf.stack([row_indices, self.action_ph], axis=1)
        actioned_outputs = tf.gather_nd(self.output, indices)

        self.loss = -tf.reduce_mean(tf.log(actioned_outputs) * self.reward_ph)

        tvars = tf.trainable_variables()
        self.gradient_phs = []
        for i, var in enumerate(tvars):
            ph = tf.placeholder(tf.float32, name='grad{}_ph'.format(i))
            self.gradient_phs.append(ph)

        self.gradients = tf.gradients(self.loss, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_phs, tvars))
