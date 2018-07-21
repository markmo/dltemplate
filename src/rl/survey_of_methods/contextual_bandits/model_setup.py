import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class ContextualBandit(object):
    """
    Here we define our contextual bandits. In this example, we are using three four-armed bandits.
    What this means is that each bandit has four arms that can be pulled. Each bandit has different
    success probabilities for each arm, and as such, requires different actions to obtain the best
    result. The pull_bandit function generates a random number from a normal distribution with a
    mean of 0. The lower the bandit number, the more likely a positive reward will be returned. We
    want our agent to learn to always choose the bandit-arm that will most often give a positive
    reward, depending on the bandit presented.
    """

    def __init__(self):
        self.state = 0

        # Currently arms 4, 2, and 1 (respectively) are the most optimal
        self.bandits = np.array([[.2, 0., -0., -5.], [.1, -5., 1., .25], [-5., 5., 5., 5.]])
        self.n_bandits = self.bandits.shape[0]
        self.n_actions = self.bandits.shape[1]

    def get_bandit(self):
        """ returns a random state for each episode """
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        return 1 if result > bandit else -1


class Agent(object):
    """
    This establishes our policy-based neural agent. It takes as input the current state,
    and returns an action. This allows the agent to take actions which are conditioned
    on the state of the environment - a critical step toward being able to solve full
    RL problems.

    The agent uses a single set of weights, within which each value is an estimate of
    the value of the return from choosing a particular arm given a bandit. We use a
    policy gradient method to update the agent by moving the value for the selected
    action towards the received reward.
    """

    def __init__(self, learning_rate, state_dim, n_actions):
        # These lines established the feed-forward part of the network.
        # The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_one_hot = slim.one_hot_encoding(self.state_in, state_dim)
        output = slim.fully_connected(state_in_one_hot, n_actions,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer(),
                                      biases_initializer=None)
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        self.reward_ph = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_ph = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_ph, [1])
        self.loss = -tf.log(self.responsible_weight) * self.reward_ph
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update_op = optimizer.minimize(self.loss)
