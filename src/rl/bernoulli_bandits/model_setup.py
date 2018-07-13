from abc import ABCMeta, abstractmethod
import numpy as np


class BernoulliBandit(object):

    def __init__(self, n_actions=5):
        self._probs = np.random.random(n_actions)

    @property
    def action_count(self):
        return len(self._probs)

    def pull(self, action):
        if np.random.random() > self._probs[action]:
            return 0.0

        return 1.0

    def optimal_reward(self):
        """ Used for regret calculation """
        return np.max(self._probs)

    def step(self):
        """ Used in non-stationary version """
        pass

    def reset(self):
        """ Used in non-stationary version """


# noinspection PyAttributeOutsideInit
class AbstractAgent(metaclass=ABCMeta):

    def init_actions(self, n_actions):
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._total_pulls = 0

    @abstractmethod
    def get_action(self):
        """
        Get current best action

        :return: (int)
        """
        pass

    def update(self, action, reward):
        """
        Observe reward from action and update agent's internal parameters

        :param action: (int)
        :param reward: (int)
        :return:
        """
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

    @property
    def name(self):
        return self.__class__.__name__


class RandomAgent(AbstractAgent):

    def get_action(self):
        return np.random.randint(0, len(self._successes))


class EpsilonGreedyAgent(AbstractAgent):
    """
    for t = 1, 2, ... do
        for k=1, ..., K do
            θ̂ k ← αk / (αk + βk)
        end for
        xt ← argmax_k θ̂  with probability 1−ϵ or random action with probability ϵ
        Apply  xt and observe rt
        (α_xt, β_xt) ← (α_xt, β_xt) + (r_t, 1 − r_t)
    end for
    """

    def __init__(self, epsilon=0.01):
        super().__init__()
        self._epsilon = epsilon

    def get_action(self):
        n_actions = self._successes + self._failures

        with np.errstate(invalid='ignore'):
            p = self._successes / n_actions

        # explore else exploit
        if np.random.random() < self._epsilon:
            return np.random.randint(0, len(self._successes))
        else:
            return np.argmax(p)

    @property
    def name(self):
        return self.__class__.__name__ + '(epsilon={})'.format(self._epsilon)


class UCBAgent(AbstractAgent):
    """
    Optimism in the Face of Uncertainty - UCB1 (Upper Confidence Bound)

    Epsilon-greedy strategy has no preference for actions. It would be better to select
    among actions that are uncertain or have potential to be optimal. One can come up
    with idea of index for each action that represents optimality and uncertainty at
    the same time. One efficient way to do it is to use the UCB1 algorithm:

    for t = 1, 2, ... do
        for k = 1, ..., K do
            w_k ← α_k / (α_k + β_k) + √(2log t / (α_k + β_k))
        end for
        x_t ← argmax_k(w)
        Apply x_t and observe r_t
        (α_xt, β_xt) ← (α_xt, β_xt) + (r_t, 1 − r_t)
    end for

    Note: in practice, one can multiply √(2log t / (α_k + β_k)) by some tunable parameter
    to regulate the agent's optimism and willingness to abandon unpromising actions.

    More versions and optimality analysis at https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    """

    def get_action(self):
        n_actions = self._successes + self._failures

        with np.errstate(divide='ignore', invalid='ignore'):
            ucb = np.sqrt(2 * np.log10(self._total_pulls) / n_actions)
            p = self._successes / n_actions + ucb

        return np.argmax(p)

    @property
    def name(self):
        return self.__class__.__name__


class ThompsonSamplingAgent(AbstractAgent):
    """
    The UCB1 algorithm does not take into account actual distribution of rewards.
    If we know the distribution - we can do much better by using Thompson sampling:

    for t = 1, 2, ... do
        for k = 1, ..., K do
            Sample θ̂ k ∼ beta(αk, βk)
        end for
        x_t ← argmax_kθ̂
        Apply x_t and observe r_t
        (αxt, βxt) ← (αxt, βxt) + (r_t, 1 − r_t)
    end for

    More on Thompson Sampling: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    """

    def get_action(self):
        p = np.random.beta(self._successes + 1, self._failures + 1)
        return np.argmax(p)

    @property
    def name(self):
        return self.__class__.__name__
