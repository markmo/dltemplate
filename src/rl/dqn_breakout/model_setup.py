from gym.core import ObservationWrapper
from gym.spaces import Box
from keras.layers import Conv2D, Dense, Flatten
import keras
import numpy as np
from scipy.misc import imresize
import tensorflow as tf


class DQNAgent(object):

    def __init__(self, name, state_shape, n_actions, epsilon=0, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            network = keras.models.Sequential()
            network.add(Conv2D(16, (3, 3), strides=2, activation='relu', input_shape=state_shape))
            network.add(Conv2D(32, (3, 3), strides=2, activation='relu'))
            network.add(Conv2D(64, (3, 3), strides=2, activation='relu'))
            network.add(Flatten())
            network.add(Dense(256, activation='relu'))
            network.add(Dense(n_actions, activation='linear'))
            self.network = network
            self.n_actions = n_actions
            self.state_t = tf.placeholder('float32', [None,] + list(state_shape))
            self.q_values_t = self.get_symbolic_q_values(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_symbolic_q_values(self, state_t):
        """ Takes agent's observation, returns Q-values. Both are tf tensors. """
        q_values = self.network(state_t)

        assert tf.is_numeric_tensor(q_values) and q_values.shape.ndims == 2, \
            'Please return 2D tf tensor of Q-values, got %s' % repr(q_values)
        assert int(q_values.shape[1]) == self.n_actions

        return q_values

    def get_q_values(self, state_t):
        """ Same as symbolic step except operates on numpy arrays """
        sess = tf.get_default_session()
        return sess.run(self.q_values_t, {self.state_t: state_t})

    def sample_actions(self, q_values):
        """ Pick actions given Q-values. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = q_values.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = q_values.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


class PreprocessAtariImage(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.img_size = (64, 64)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def _observation(self, img):
        img = img[34:-16, :, :]
        img = imresize(img, self.img_size)
        img = img.mean(-1, keepdims=True)
        img = img.astype('float32')
        img /= 255.
        return img
