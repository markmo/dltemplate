from collections import defaultdict
import numpy as np
import random


class QLearningAgent(object):

    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent

        Based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

        :param alpha: learning rate
        :param epsilon: exploration probability
        :param discount: discount rate - gamma
        :param get_legal_actions: {state, hashable -> list of actions, each is hashable}
        """
        self.get_legal_actions = get_legal_actions
        self._q_values = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_q_value(self, state, action):
        """ Returns Q(state, action) """
        return self._q_values[state][action]

    def set_q_value(self, state, action, value):
        """ Sets the Q value for (state, action) to the given value """
        self._q_values[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.

        :param state:
        :return:
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.

        return np.max([self.get_q_value(state, action) for action in possible_actions])

    def update(self, state, action, reward, next_state):
        """
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        """
        gamma = self.discount
        learning_rate = self.alpha
        q_value = ((1 - learning_rate) * self.get_q_value(state, action) +
                   learning_rate * (reward + gamma + self.get_value(next_state)))
        self.set_q_value(state, action, q_value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).

        :param state:
        :return:
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        q_values = [self.get_q_value(state, action) for action in possible_actions]
        return possible_actions[np.argmax(q_values)]

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        Take a random action with probability self.epsilon, otherwise take the
        best policy action (`self.get_best_action`).

        :param state:
        :return:
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(possible_actions)
        else:
            action = self.get_best_action(state)

        return action
