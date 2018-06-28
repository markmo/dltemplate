from gym.core import Wrapper
from gym.spaces.box import Box
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

try:
    # noinspection PyUnresolvedReferences
    from IPython.display import display

    # noinspection PyUnresolvedReferences
    from graphviz import Digraph

    import graphviz
    has_graphviz = True
except ImportError:
    has_graphviz = False


class FrameBuffer(Wrapper):
    """
    Our agent can only process one observation at a time, so we need to make
    sure it contains enough information to find optimal actions. For instance,
    for the agent to react to moving objects, it must be able to sense the
    object's velocity.
    """
    def __init__(self, env, n_frames=4, dim_order='tensorflow'):
        super().__init__(env)
        self.dim_order = dim_order
        if dim_order == 'tensorflow':
            height, width, n_channels = env.observation_space.shape
            obs_shape = [height, width, n_channels * n_frames]
        elif dim_order == 'pytorch':
            n_channels, height, width = env.observation_space.shape
            obs_shape = [n_channels * n_frames, height, width]
        else:
            raise ValueError("dim_order should be 'tensorflow' or 'pytorch', got {}".format(dim_order))
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

    def step(self, action):
        """ plays breakout for one step, returns frame buffer """
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        if self.dim_order == 'tensorflow':
            offset = self.env.observation_space.shape[-1]
            axis = -1
            cropped_framebuffer = self.framebuffer[:, :, :-offset]
        elif self.dim_order == 'pytorch':
            offset = self.env.observation_space.shape[0]
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset]
        else:
            raise ValueError("dim_order should be 'tensorflow' or 'pytorch', got {}".format(self.dim_order))

        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)


class MDP(object):
    """
    Defines a Markov Decision Process (MDP),
    compatible with OpenAI Gym environment.
    """
    def __init__(self, transition_probs, rewards, initial_state=None):
        """
        States and actions can be anything you can use as dict keys, but we
        recommend that you use strings or integers.

        Here's an example from the MDP depicted on http://bit.ly/2jrNHNr

        transition_probs = {
          's0': {
            'a0': {'s0': 0.5, 's2': 0.5},
            'a1': {'s2': 1}
          },
          's1': {
            'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
            'a1': {'s1': 0.95, 's2': 0.05}
          },
          's2': {
            'a0': {'s0': 0.4, 's1': 0.6},
            'a1': {'s0': 0.3, 's1': 0.3, 's2':0.4}
          }
        }
        rewards = {
            's1': {'a0': {'s0': +5}},
            's2': {'a1': {'s0': -1}}
        }

        :param transition_probs: transition_probs[s][a][s_next] = P(s_next | s, a)
               A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> prob].
               For each state and action, probabilities of next states should sum to 1.
               If a state has no actions available, it is considered terminal.
        :param rewards: rewards[s][a][s_next] = r(s,a,s')
               A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> reward].
               The reward for anything not mentioned here is zero.
        :param initial_state: a state where agent starts or a callable() -> state
               By default, initial state is random.

        """
        _check_param_consistency(transition_probs, rewards)
        self.transition_probs = transition_probs
        self._rewards = rewards
        self.initial_state = initial_state
        self.n_states = len(transition_probs)
        self._current_state = None
        self.reset()

    def get_all_states(self):
        return tuple(self.transition_probs.keys())

    def get_possible_actions(self, state):
        return tuple(self.transition_probs.get(state, {}).keys())

    def is_terminal(self, state):
        return len(self.get_possible_actions(state)) == 0

    def get_next_states(self, state, action):
        """
        Return a dict of {next_state1: P(next_state1|state, action), next_state2: ...}

        :param state:
        :param action:
        :return:
        """
        assert action in self.get_possible_actions(state),\
            'cannot do action %s from state %s' % (action, state)

        return self.transition_probs[state][action]

    def get_transition_prob(self, state, action, next_state):
        """
        Return P(next_state|state, action)

        :param state:
        :param action:
        :param next_state:
        :return:
        """
        return self.get_next_states(state, action).get(next_state, 0.)

    def get_reward(self, state, action, next_state):
        """
        Return the reward you get for taking action in state and landing on next_state

        :param state:
        :param action:
        :param next_state:
        :return:
        """
        assert action in self.get_possible_actions(state),\
            'cannot do action %s from state %s' % (action, state)

        return self._rewards.get(state, {}).get(action, {}).get(next_state, 0.)

    def reset(self):
        """
        Reset the game, and return the initial state

        :return:
        """
        if self.initial_state is None:
            self._current_state = random.choice(tuple(self.transition_probs.keys()))
        elif self.initial_state in self.transition_probs:
            self._current_state = self.initial_state
        elif callable(self.initial_state):
            self._current_state = self.initial_state()
        else:
            raise ValueError('initial state %s should be either '
                             'a state or a function() -> state' % self.initial_state)

        return self._current_state

    def step(self, action):
        """
        take action, return next_state, reward, is_done, empty_info

        :param action:
        :return:
        """
        possible_states, probs = zip(*self.get_next_states(self._current_state, action).items())
        next_state = weighted_choice(possible_states, p=probs)
        reward = self.get_reward(self._current_state, action, next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state

        return next_state, reward, is_done, {}

    def render(self):
        print('Currently at %s' % self._current_state)


class ReplayBuffer(object):

    def __init__(self, size):
        """

        :param size: (int) Max number of transitions to store in the buffer. When the buffer
                     overflows, the old memories are dropped.
        """
        self.storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self.storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxs):
        """

        :param idxs:
        :return:
            obsvs: (np.array) batch of observations
            actions: (np.array) batch of actions executed given observations
            rewards: (np.array) rewards received as a result of executing actions
            next_obsvs: (np.array) next set of observations seen after executing actions
            done_mask: (np.array) done_mask[i] = 1 if executing actions[i] resulted in
                       the end of an episode and 0 otherwise.
        """
        obsvs, actions, rewards, next_obsvs, done_mask = [], [], [], [], []
        for i in idxs:
            obs, action, reward, next_obs, done = self.storage[i]
            obsvs.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_obsvs.append(np.array(next_obs, copy=False))
            done_mask.append(done)

        return (np.array(obsvs),
                np.array(actions),
                np.array(rewards),
                np.array(next_obsvs),
                np.array(done_mask))

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        :param batch_size: (int) number of transitions to sample
        :return:
        """
        idxs = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxs)


def draw_policy(mdp, state_values, gamma=0.9):
    plt.figure(figsize=(3, 3))
    h, w = mdp.desc.shape
    states = sorted(mdp.get_all_states())
    v_ = np.array([state_values[s] for s in states])
    pi = {s: get_optimal_action(mdp, state_values, s, gamma) for s in states}
    plt.imshow(v_.reshape(w, h), cmap='gray', interpolation='none', clim=(0, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(h) - .5)
    ax.set_yticks(np.arange(w) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    a2uv = {'left': (-1, 0), 'down': (0, -1), 'right': (1, 0), 'up': (-1, 0)}
    for y in range(h):
        for x in range(w):
            plt.text(x, y, str(mdp.desc[y, x].item()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            a = pi[y, x]
            if a is None:
                continue

            u, v = a2uv[a]
            plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.1, head_length=0.1)

    plt.grid(color='b', lw=2, ls='-')
    plt.show()


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """
    Plays n_games full games. If greedy, picks actions as argmax(q_values).

    :param env:
    :param agent:
    :param n_games:
    :param greedy:
    :param t_max:
    :return: mean reward
    """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            q_values = agent.get_q_values([s])
            action = q_values.argmax(axis=-1)[0] if greedy else agent.sample_actions(q_values)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)

    return np.mean(rewards)


def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) """
    q = 0
    for s_next, prob in mdp.get_next_states(state, action).items():
        q += prob * (mdp.get_reward(state, action, s_next) + gamma * state_values[s_next])

    return q


def get_optimal_action(mdp, state_values, state, gamma=0.9):
    """ Finds optimal action. """
    if mdp.is_terminal(state):
        return None

    actions = mdp.get_possible_actions(state)
    values = [get_action_value(mdp, state_values, state, a, gamma) for a in actions]

    return actions[np.argmax(values)]


def get_optimal_action_for_plot(mdp, state_values, state, gamma=0.9):
    if mdp.is_terminal(state):
        return None

    next_actions = mdp.get_possible_actions(state)
    q_values = [get_action_value(mdp, state_values, state, action, gamma) for action in next_actions]
    optimal_action = next_actions[np.argmax(q_values)]

    return optimal_action


def load_weights_into_target_network(agent, target_network):
    """
    Assign target_network.weights variables to their respective agent.weights
    values.

    The "target network" is a copy of neural network weights to be used for
    reference Q-values.

    The network itself is an exact copy of the agent network, except its
    parameters are not trained. Instead, they are moved here from the
    agent's actual network every so often.
    """
    assignments = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assignments.append(tf.assign(w_target, w_agent, validate_shape=True))

    tf.get_default_session().run(assignments)


def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s, a, r, s', done)
    to replay buffer.

    Whenever game ends, add record with done=True and reset the game.

    It is guaranteed that env has done=False when passed to this function.

    :param agent:
    :param env:
    :param exp_replay:
    :param n_steps:
    :return: sum of rewards over time
    """
    # initial state
    s = env.framebuffer

    # play the game for n_steps as per instructions above
    reward = 0.0
    for t in range(n_steps):
        # get agent to pick action given state s
        q_values = agent.get_q_values([s])
        action = agent.sample_actions(q_values)[0]
        next_s, r, done, _ = env.step(action)

        # add to replay buffer
        exp_replay.add(s, action, r, next_s, done)
        reward += r
        if done:
            s = env.reset()
        else:
            s = next_s

    return reward


# noinspection SpellCheckingInspection
def plot_graph(mdp, graph_size='10,10', s_node_size='1,5', a_node_size='0,5', rankdir='LR'):
    """
    Function for drawing pretty MDP graph with graphviz library.

    Requirements:
    * graphviz: https://www.graphviz.org/
      For Ubuntu users: sudo apt-get install graphviz
    * Python library for graphviz
      For pip users: pip install graphviz

    :param mdp:
    :param graph_size: size of graph plot
    :param s_node_size: size of state nodes
    :param a_node_size: size of action nodes
    :param rankdir: order for drawing
    :return: {Digraph} graphviz dot object
    """
    s_node_attrs = {
        'shape': 'doublecircle',
        'color': 'lightgreen',
        'style': 'filled',
        'width': str(s_node_size),
        'height': str(s_node_size),
        'fontname': 'Arial',
        'fontsize': '24'
    }
    a_node_attrs = {
        'shape': 'circle',
        'color': 'lightpink',
        'style': 'filled',
        'width': str(a_node_size),
        'height': str(a_node_size),
        'fontname': 'Arial',
        'fontsize': '20'
    }
    s_a_edge_attrs = {
        'style': 'bold',
        'color': 'red',
        'ratio': 'auto'
    }
    a_s_edge_attrs = {
        'style': 'dashed',
        'color': 'blue',
        'ratio': 'auto',
        'fontname': 'Arial',
        'fontsize': '16'
    }
    graph = Digraph(name='MDP')
    graph.attr(rankdir=rankdir, size=graph_size)
    for state_node in mdp.transition_probs:
        graph.node(state_node, **s_node_attrs)
        for possible_action in mdp.get_possible_actions(state_node):
            action_node = state_node + '-' + possible_action
            graph.node(action_node, label=str(possible_action), **a_node_attrs)
            graph.edge(state_node, state_node + '-' + possible_action, **s_a_edge_attrs)
            for posible_next_state in mdp.get_next_states(state_node, possible_action):
                prob = mdp.get_transition_prob(state_node, possible_action, posible_next_state)
                reward = mdp.get_reward(state_node, possible_action, posible_next_state)
                if reward != 0:
                    label_a_s_edge = 'p = ' + str(prob) + '  ' + 'reward =' + str(reward)
                else:
                    label_a_s_edge = 'p = ' + str(prob)

                graph.edge(action_node, posible_next_state, label=label_a_s_edge, **a_s_edge_attrs)

    return graph


# noinspection PyTypeChecker
def plot_graph_with_state_values(mdp, state_values):
    graph = plot_graph(mdp)
    for state_node in mdp.transition_probs:
        value = state_values[state_node]
        graph.node(state_node, label=str(state_node) + '\n' + 'V =' + str(value)[:4])

    return display(graph)


# noinspection PyTypeChecker,SpellCheckingInspection
def plot_graph_optimal_strategy_and_state_values(mdp, state_values, gamma=0.8):
    opt_s_a_edge_attrs = {
        'style': 'bold',
        'color': 'green',
        'ratio': 'auto',
        'penwidth': '6'
    }
    graph = plot_graph(mdp)
    for state_node in mdp.transition_probs:
        value = state_values[state_node]
        graph.node(state_node, label=str(state_node) + '\n' + 'V =' + str(value)[:4])
        for action in mdp.get_possible_actions(state_node):
            if action == get_optimal_action_for_plot(mdp, state_values, state_node, gamma):
                graph.edge(state_node, state_node + "-" + action, **opt_s_a_edge_attrs)

    return display(graph)


def sample_batch(exp_replay, batch_size, obsvs_ph, actions_ph, rewards_ph, next_obsvs_ph, is_done_ph):
    obsvs_batch, actions_batch, rewards_batch, next_obsvs_batch, is_done_batch = exp_replay.sample(batch_size)
    return {
        obsvs_ph: obsvs_batch,
        actions_ph: actions_batch,
        rewards_ph: rewards_batch,
        next_obsvs_ph: next_obsvs_batch,
        is_done_ph: is_done_batch
    }


def weighted_choice(v, p):
    total = sum(p)
    r = random.uniform(0, total)
    up_to = 0
    for c, w in zip(v, p):
        if up_to + w >= r:
            return c

        up_to += w

    assert False, "Shouldn't get here"


def _check_param_consistency(transition_probs, rewards):
    for state in transition_probs:
        assert isinstance(transition_probs[state], dict),\
            'transition_probs for %s should be a dictionary, '\
            'but is instead %s' % (state, type(transition_probs[state]))

        for action in transition_probs[state]:
            assert isinstance(transition_probs[state][action], dict),\
                'transition_probs for %s, %s should be a dictionary, '\
                'but is instead %s' % (state, action, type(transition_probs[state, action]))

            next_state_probs = transition_probs[state][action]
            assert len(next_state_probs) != 0,\
                'from state %s action %s leads to no next states' % (state, action)

            sum_probs = sum(next_state_probs.values())
            assert abs(sum_probs - 1) <= 1e-10,\
                'next state probabilities for state %s action %s '\
                'add up to %f (should be 1)' % (state, action, sum_probs)

    for state in rewards:
        assert isinstance(rewards[state], dict),\
            'rewards for %s should be a dictionary, '\
            'but is instead %s' % (state, type(transition_probs[state]))

        for action in rewards[state]:
            assert isinstance(rewards[state][action], dict),\
                'rewards for %s, %s should be a dictionary, '\
                'but is instead %s' % (state, action, type(transition_probs[state, action]))

    assert None not in transition_probs, 'please do not use None as a state identifier'
    assert None not in rewards, 'please do not use None as an action identifier'
