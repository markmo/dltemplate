from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
import tensorflow as tf


class Agent(object):

    def __init__(self, name, state_shape, n_actions, reuse=False):
        """ A simple actor-critic agent """
        with tf.variable_scope(name, reuse=reuse):
            inp = Input(state_shape)
            conv1 = Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu')(inp)
            conv2 = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu')(conv1)
            dense1 = Dense(256)(Flatten()(conv2))
            dense2 = Dense(32)(dense1)

            out1 = Dense(n_actions)(dense2)
            out2 = Dense(1)(dense2)

            self.network = Model(inp, [out1, out2])

            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.agent_outputs = self.symbolic_step(self.state_t)

    def symbolic_step(self, state_t):
        """
        Takes agent's previous step and observation, returns next state and
        whatever it needs to learn (tf tensors).
        :param state_t:
        :return:
        """
        logits, state_value = self.network(state_t)
        state_value = state_value[:, 0]

        assert tf.is_numeric_tensor(state_value) and state_value.shape.ndims == 1, \
            'please return 1D tf tensor of state values [you got %s]' % repr(state_value)
        assert tf.is_numeric_tensor(logits) and logits.shape.ndims == 2, \
            'please return 2d tf tensor of logits [you got %s]' % repr(logits)

        # hint: if you triggered state_values assert with your shape being [None, 1],
        # just select [:, 0]-th element of state values as new state values

        return logits, state_value

    def step(self, sess, state_t):
        """ Same as symbolic_step except operates on numpy arrays """
        return sess.run(self.agent_outputs, {self.state_t: state_t})


class ActorCritic(object):

    def __init__(self, obs_shape, n_actions, agent, gamma=0.99, learning_rate=1e-4):
        self.states_ph = tf.placeholder('float32', [None, ] + list(obs_shape))
        self.next_states_ph = tf.placeholder('float32', [None, ] + list(obs_shape))
        self.actions_ph = tf.placeholder('int32', (None,))
        self.rewards_ph = tf.placeholder('float32', (None,))
        self.is_done_ph = tf.placeholder('float32', (None,))

        # logits[n_envs, n_actions] and state_values[n_envs, n_actions]
        logits, state_values = agent.symbolic_step(self.states_ph)
        next_logits, next_state_values = agent.symbolic_step(self.next_states_ph)
        next_state_values = next_state_values * (1 - self.is_done_ph)

        # probabilities and log-probabilities for all actions
        probs = tf.nn.softmax(logits)  # [n_envs, n_actions]
        log_probs = tf.nn.log_softmax(logits)  # [n_envs, n_actions]

        # log-probabilities only for agent's chosen actions
        logp_actions = tf.reduce_mean(log_probs * tf.one_hot(self.actions_ph, n_actions), axis=-1)  # [n_envs]

        # compute advantage using rewards_ph, state_values, and next_state_values
        self.advantage = self.rewards_ph + gamma * (next_state_values - state_values)

        assert self.advantage.shape.ndims == 1, 'please compute advantage for each sample, vector of shape [n_envs,]'

        self.entropy = -tf.reduce_sum(probs * log_probs, 1)

        assert self.entropy.shape.ndims == 1, 'please compute pointwise entropy vector of shape [n_envs,]'

        self.actor_loss = -tf.reduce_mean(logp_actions * tf.stop_gradient(self.advantage)) \
                          - 0.001 * tf.reduce_mean(self.entropy)

        # compute target state values using temporal difference formula
        target_state_values = self.rewards_ph + gamma * next_state_values

        self.critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2)

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss + self.critic_loss)
