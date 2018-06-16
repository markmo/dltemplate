import numpy as np
from rl.qlearning.qlearning_agent import QLearningAgent


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of its q-learning functions to implement
    Expected Value SARSA.
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under epsilon-greedy policy:

            V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        :param state:
        :return:
        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)
        n_actions = len(possible_actions)
        if n_actions == 0:
            return 0.

        q_values = [self.get_q_value(state, action) for action in possible_actions]
        best_action_idx = np.argmax(q_values)
        expected_value = 0.
        for i in range(n_actions):
            if i == best_action_idx:
                expected_value += (1 - epsilon + epsilon / n_actions) * q_values[i]
            else:
                expected_value += (epsilon / n_actions) * q_values[i]

        return expected_value
