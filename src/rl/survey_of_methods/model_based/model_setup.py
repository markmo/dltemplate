import tensorflow as tf


class PolicyNetwork(object):

    def __init__(self, obs_dim, n_hidden, learning_rate=1e-2):
        self.observations = tf.placeholder(tf.float32, [None, obs_dim], name='input_X')
        w1 = tf.get_variable('W1', shape=[obs_dim, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.observations, w1))
        w2 = tf.get_variable('W2', shape=[n_hidden, 1], initializer=tf.contrib.layers.xavier_initializer())
        score = tf.matmul(layer1, w2)
        self.probability = tf.nn.sigmoid(score)

        self.tvars = tf.trainable_variables()
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        self.advantage = tf.placeholder(tf.float32, name='reward_signal')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.w1_grad = tf.placeholder(tf.float32, name='batch_grad1')
        self.w2_grad = tf.placeholder(tf.float32, name='batch_grad2')
        batch_grads = [self.w1_grad, self.w2_grad]

        loglik = tf.log(self.input_y * (self.input_y - self.probability) +
                        (1 - self.input_y) * (self.input_y + self.probability))
        loss = -tf.reduce_mean(loglik * self.advantage)

        self.new_grads = tf.gradients(loss, self.tvars)
        self.update_grads = optimizer.apply_gradients(zip(batch_grads, self.tvars))


class ModelNetwork(object):
    """ Predicts the next observation, reward, and done state from the current state and action. """

    def __init__(self, n_hidden, learning_rate=1e-2):
        self.input_data = tf.placeholder(tf.float32, [None, 5])

        # noinspection SpellCheckingInspection
        with tf.variable_scope('rnnlm'):
            self.softmax_w = tf.get_variable('softmax_w', [n_hidden, 50])
            self.softmax_b = tf.get_variable('softmax_b', [50])

        self.prev_state = tf.placeholder(tf.float32, [None, 5], name='prev_state')
        w1 = tf.get_variable('W1_model', shape=[5, n_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.zeros([n_hidden]), name='b1_model')
        layer1 = tf.nn.relu(tf.matmul(self.prev_state, w1) + b1)
        w2 = tf.get_variable('W2_model', shape=[n_hidden, n_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.zeros([n_hidden]), name='b2_model')
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

        w_obs = tf.get_variable('w_obs', shape=[n_hidden, 4],
                                initializer=tf.contrib.layers.xavier_initializer())
        w_reward = tf.get_variable('w_reward', shape=[n_hidden, 1],
                                   initializer=tf.contrib.layers.xavier_initializer())
        w_done = tf.get_variable('w_done', shape=[n_hidden, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())

        b_obs = tf.Variable(tf.zeros([4]), name='b_obs')
        b_reward = tf.Variable(tf.zeros([1]), name='b_reward')
        b_done = tf.Variable(tf.ones([1]), name='b_done')

        pred_obs = tf.matmul(layer2, w_obs, name='pred_obs') + b_obs
        pred_reward = tf.matmul(layer2, w_reward, name='pred_reward') + b_reward
        pred_done = tf.sigmoid(tf.matmul(layer2, w_done, name='pred_done') + b_done)

        self.true_obs = tf.placeholder(tf.float32, [None, 4], name='true_obs')
        self.true_reward = tf.placeholder(tf.float32, [None, 1], name='true_reward')
        self.true_done = tf.placeholder(tf.float32, [None, 1], name='true_done')

        self.pred_state = tf.concat([pred_obs, pred_reward, pred_done], axis=1)

        loss_obs = tf.square(self.true_obs - pred_obs)
        loss_reward = tf.square(self.true_reward - pred_reward)
        loss_done = -tf.log(tf.multiply(pred_done, self.true_done) + tf.multiply(1 - pred_done, 1 - self.true_done))

        self.loss = tf.reduce_mean(loss_obs + loss_reward + loss_done)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update_op = optimizer.minimize(self.loss)
