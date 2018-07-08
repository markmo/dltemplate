from gym.core import Env
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from rl.util import WithSnapshots
from typing import Optional


class Node(object):
    """ A tree node for MCTS """

    parent = None  # parent node
    value_sum = 0.  # sum of state values from all visits (numerator)
    n_visited = 0.  # counter of visits (denominator)

    def __init__(self, env: WithSnapshots, n_actions: int, parent: Optional['Node'], action: Optional[int]) -> None:
        """
        Creates an empty node with no children, by committing an action and
        recording the outcome.

        :param env:
        :param n_actions:
        :param parent: parent node
        :param action: action to commit from parent node
        """
        self.env = env
        self.n_actions = n_actions
        self.parent = parent
        self.action = action
        self.children = set()

        # get action outcome and save it
        if parent and action is not None:
            res = env.get_result(parent.snapshot, action)
            self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def get_mean_value(self) -> float:
        return self.value_sum / self.n_visited if self.n_visited != 0 else 1.

    def ucb_score(self, scale: float = 10, max_value: float = 1e100) -> float:
        """
        Computes UCB1 upper bound using current value and visit counts
        for node and its parent.

        :param scale: multiples upper bound. From Hoeffding Inequality, assumes reward range to be [0, scale].
        :param max_value: a value that represents infinity for unvisited nodes.
        :return:
        """
        if self.n_visited == 0:
            return max_value

        # compute UCB1 additive component to be added to mean value
        # use `self.parent.n_visited` for n times node was considered,
        # and `self.n_visited` for n times it was visited.
        upper_confidence_bound = np.sqrt(2 * np.log(self.parent.n_visited) / self.n_visited)
        return self.get_mean_value() + scale * upper_confidence_bound

    def select_best_leaf(self) -> 'Node':
        """
        Picks the leaf with highest priority to expand, by recursively picking
        nodes with the best UCB1 score until it reaches the leaf.

        :return:
        """
        if self.is_leaf():
            return self

        best_child = max([(child.ucb_score(), child) for child in self.children], key=lambda x: x[0])[1]
        return best_child.select_best_leaf()

    def expand(self) -> 'Node':
        """
        Expands the current node by creating all possible child nodes.

        Then returns one of those children.

        :return:
        """
        assert not self.is_done, "Can't expand from terminal state"

        for action in range(self.n_actions):
            self.children.add(Node(self.env, self.n_actions, self, action))

        return self.select_best_leaf()

    def rollout(self, t_max: int = 10**4):
        """
        Play the game from this state to the end (done) or for t_max steps.

        On each step, pick action at random.

        :param t_max:
        :return:
        """
        # set env to the appropriate state
        self.env.load_snapshot(self.snapshot)
        done = self.is_done

        rollout_reward = 0
        while not done and t_max > 0:
            t_max -= 1
            _, r, done, _ = self.env.step(self.env.action_space.sample())
            rollout_reward += r

        return rollout_reward

    def propagate(self, child_value: float):
        """ Uses child value (sum of rewards) to update parents recursively. """
        # compute node value
        value = self.immediate_reward + child_value

        # update value_sum and n_visited
        self.value_sum += value
        self.n_visited += 1

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(value)

    def safe_delete(self) -> None:
        """ safe delete to prevent memory leak in some python versions """
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child


class Root(Node):

    def __init__(self, env: WithSnapshots, n_actions: int, snapshot: bytes, observation: np.ndarray) -> None:
        """
        Creates special node that acts like tree root.

        :param env:
        :param n_actions:
        :param snapshot: snapshot to start planning from (from `env.get_snapshot`)
        :param observation: last env observation
        """
        super().__init__(env, n_actions, None, None)

        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0.
        self.is_done = False

    @staticmethod
    def from_node(node: Node) -> 'Root':
        """ initializes node as root """
        root = Root(node.env, node.n_actions, node.snapshot, node.observation)

        copied_fields = ['value_sum', 'n_visited', 'children', 'is_done']
        for field in copied_fields:
            setattr(root, field, getattr(node, field))

        return root


def plan_mcts(root: Node, n_iters: int = 10) -> None:
    """
    Builds tree with Monte Carlo Tree Search for `n_iters` iterations.

    :param root: tree node to plan from
    :param n_iters: number of 'select, expand, simulate, propagate' loops to make
    :return:
    """
    for _ in range(n_iters):
        node = root.select_best_leaf()
        if node.is_done:
            node.propagate(0)
        else:
            node_child = node.expand()
            child_reward = node_child.rollout()
            node.propagate(child_reward)


def train(root: Root, env: Env, show: bool = False) -> None:
    total_reward = 0
    for i in count():
        best_child = max([(child.get_mean_value(), child) for child in root.children], key=lambda x: x[0])[1]
        s, r, done, _ = env.step(best_child.action)

        # show image
        if show:
            plt.title('Step %i' % i)
            plt.imshow(env.render('rgb_array'))
            plt.show()

        total_reward += r
        if done:
            print('Finished with reward =', total_reward)
            break

        # discard unrealized part of the tree [because not every child matters :(]
        for child in root.children:
            if child != best_child:
                child.safe_delete()

        # declare best child a new root
        root = Root.from_node(best_child)

        # auto expand
        if root.is_leaf():
            plan_mcts(root, n_iters=10)
