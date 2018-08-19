""" models. warning! uses Keras 1.x """
__credits__ = 'giorgio@ac.upc.edu'

import keras
from keras.activations import relu
from keras.layers import Dense, Input, merge
from keras.models import Model
from rl.routing_optimization.util import selu
import tensorflow as tf


class ActorNetwork(object):

    def __init__(self, sess, state_size, n_actions, constants):
        self._n_hidden1 = constants['n_hidden1']
        self._n_hidden2 = constants['n_hidden2']
        self._batch_size = constants['batch_size']
        self._tau = constants['tau']
        self._learning_rate = constants['lr_actor']
        self._action_type = constants['action_type']
        self._sess = sess
        if self._action_type == 'new':
            self._activation_fn = 'sigmoid'
        elif self._action_type == 'delta':
            self._activation_fn = 'tanh'

        if constants['hidden_layer_activation_fn'] == 'selu':
            self._hidden_layer_activation_fn = selu
        else:
            self._hidden_layer_activation_fn = 'relu'

        keras.backend.set_session(sess)

        # Create the model
        self.model, self.weights, self.state = self._create_actor_network(state_size, n_actions)
        self.target_model, self.target_weights, self.target_state = self._create_actor_network(state_size, n_actions)
        self.action_gradient = tf.placeholder(tf.float32, [None, n_actions])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self._learning_rate).apply_gradients(grads)
        self._sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self._sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def train_target(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()

        # TODO
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self._tau * actor_weights[i] + (1 - self._tau) * actor_target_weights[i]

        self.target_model.set_weights(actor_target_weights)

    def _create_actor_network(self, state_size, n_actions):
        s = Input([state_size], name='a_S')
        h0 = Dense(self._n_hidden1, activation=self._hidden_layer_activation_fn, init='glorot_normal', name='a_h0')(s)
        h1 = Dense(self._n_hidden2, activation=self._hidden_layer_activation_fn, init='glorot_normal', name='a_h1')(h0)

        # https://github.com/fchollet/keras/issues/374
        v = Dense(n_actions, activation=self._activation_fn, init='glorot_normal', name='a_V')(h1)
        model = Model(input=s, output=v)

        return model, model.trainable_weights, s


class CriticNetwork(object):

    def __init__(self, sess, state_size, n_actions, constants):
        self._n_hidden1 = constants['n_hidden1']
        self._n_hidden2 = constants['n_hidden2']
        self._batch_size = constants['batch_size']
        self._tau = constants['tau']
        self._learning_rate = constants['lr_critic']
        self._n_actions = n_actions
        self._sess = sess
        self._hidden_layer_activation_fn = relu
        if constants['hidden_layer_activation_fn'] == 'selu':
            self._hidden_layer_activation_fn = selu

        keras.backend.set_session(sess)

        # Create the model
        self.model, self.action, self.state = self._create_critic_network(state_size, n_actions)
        self.target_model, self.target_action, self.target_state = self._create_critic_network(state_size, n_actions)
        self.action_grads = tf.gradients(self.model.output, self.action)  # gradients for policy update
        self._sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self._sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def train_target(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()

        # TODO
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self._tau * critic_weights[i] + (1 - self._tau) * critic_target_weights[i]

        self.target_model.set_weights(critic_target_weights)

    def _create_critic_network(self, state_size, n_actions):
        s = Input([state_size], name='c_S')
        a = Input([n_actions], name='c_A')
        w1 = Dense(self._n_hidden1, activation=self._hidden_layer_activation_fn, init='glorot_normal', name='c_w1')(s)
        a1 = Dense(self._n_hidden2, activation='linear', init='glorot_normal', name='c_a1')(a)
        h1 = Dense(self._n_hidden2, activation='linear', init='glorot_normal', name='c_h1')(w1)
        h2 = merge([h1, a1], mode='sum', name='c_h2')
        h3 = Dense(self._n_hidden2, activation=self._hidden_layer_activation_fn, init='glorot_normal', name='c_h3')(h2)
        v = Dense(n_actions, activation='linear', init='glorot_normal', name='c_V')(h3)
        model = Model(input=[s, a], output=v)
        optimizer = keras.optimizers.Adam(lr=self._learning_rate)
        model.compile(loss='mse', optimizer=optimizer)

        return model, a, s
