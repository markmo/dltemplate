from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf


class PolicyEstimator(object):

    def __init__(self, state_dim, n_actions, scope='policy_estimator'):
        with tf.variable_scope(scope):
            self.states = tf.placeholder('float32', (None,) + state_dim, name='states')
            self.actions = tf.placeholder('int32', name='action_ids')
            self.cum_rewards = tf.placeholder('float32', name='cumulative_rewards')

            network = Sequential()
            network.add(Dense(32, activation='relu', input_shape=state_dim))
            network.add(Dense(32, activation='relu'))
            network.add(Dense(n_actions, activation='linear'))

        logits = network(self.states)

        self.policy = tf.nn.softmax(logits)
        log_policy = tf.nn.log_softmax(logits)

        idxs = tf.stack([tf.range(tf.shape(log_policy)[0]), self.actions], axis=-1)
        log_policy_for_actions = tf.gather_nd(log_policy, idxs)
        j = tf.reduce_mean(log_policy_for_actions * self.cum_rewards)

        # regularize with entropy
        entropy = -tf.reduce_sum(self.policy * log_policy, 1, name='entropy')

        # all network weights
        all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Weight updates. Maximizing J is same as minimizing -J. Adding negative entropy.
        loss = -j - 0.1 * entropy

        self.update_op = tf.train.AdamOptimizer().minimize(loss, var_list=all_weights)

    def get_action_probs(self, state, sess):
        return self.policy.eval({self.states: [state]}, session=sess)[0]

    def update(self, states, actions, cum_rewards, sess):
        self.update_op.run({
            self.states: states,
            self.actions: actions,
            self.cum_rewards: cum_rewards
        }, session=sess)
