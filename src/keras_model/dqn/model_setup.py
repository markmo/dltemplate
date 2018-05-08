from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import random


class DQNAgent(object):

    def __init__(self, state_size, action_size, constants):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = constants['gamma']
        self.epsilon = constants['epsilon']
        self.epsilon_decay = constants['epsilon_decay']
        self.epsilon_min = constants['epsilon_min']
        self.learning_rate = constants['learning_rate']
        self.n_hidden_units = constants['n_hidden_units']
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.n_hidden_units, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.n_hidden_units, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # explore or exploit?
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma + np.argmax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
