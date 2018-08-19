""" OpenAI-compatible Cell Simulator """
__credits__ = 'https://bitbucket.org/sandeep_chinchali/'

from gym import Env, spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from rl.routing_optimization.environments.util import get_burst_prob_actions, get_simplified_reward, report_rewards


class SimpleCellSimulator(Env):
    """
    Simple Cell Simulator implementing OpenAI interface,
    with dynamics: s' = s + a + noise
    """

    def __init__(self, constants):
        self._print_mode = constants['print_mode']
        self._deterministic_reset_mode = constants['deterministic_reset_mode']
        self._continuous_action_mode = constants['continuous_action_mode']
        self._reward_params = constants['reward_params']

        # For discrete actions, map an action [1..N] to a burst_prob [0..1]
        self._burst_prob_params = constants['burst_prob_params']

        # [C, A, N, E] = [0, 0, 0, 0]
        self._min_state_vector = constants['min_state_vector']

        # [C, A, N, E] = [1, 1, 1, 1]
        self._max_state_vector = constants['max_state_vector']

        # Value to reset state, such as min_state_vector
        self._reset_state_vector = constants['reset_state_vector']
        self._throughput_var = constants['throughput_var']

        # If we have very low throughput for n_last_entries after min_iterations_before_done
        # steps of the simulation, send a done and reset
        self._min_iterations_before_done = constants['min_iterations_before_done']
        self._n_last_entries = constants['n_last_entries']
        self._bad_throughput_threshold = constants['bad_throughput_threshold']

        # Does reward have [K - B'] penalty for throughput before hard limit of K?
        self._hard_throughput_limit_flag = constants['hard_throughput_limit_flag']

        # A DataFrame of (state, action, reward) history for logging
        self._reward_history = pd.DataFrame()
        self._iteration_index = 0

        # Batch = Episode
        self._batch_number = 0

        # Map discrete or continuous actions to bursts
        if self._continuous_action_mode:
            self._action_space = spaces.Box(low=self._burst_prob_params['min_burst_prob'],
                                            high=self._burst_prob_params['max_burst_prob'],
                                            shape=(1, ))
        else:
            # discrete actions
            self._burst_prob_params = get_burst_prob_actions(self._burst_prob_params)
            self._action_space = self._burst_prob_params['action_space']

        # state space
        self._observation_space = spaces.Box(low=self._min_state_vector, high=self._max_state_vector)

        if self._print_mode:
            print('Action space:', self._action_space)
            print('Observation space:', self._observation_space)

        self.seed()
        self.state = None
        self.reward = None
        self.action_taken = None
        self._random = np.random.RandomState()

    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ Given (s, a) return s', reward, done """
        # Convert action to a burst_prob [0..1]
        if self._continuous_action_mode:
            burst_prob = action
        else:
            burst_prob = self._burst_prob_params['action_to_burst_prob']

        # state is c - collision
        s = self.state
        noise = np.random.normal(0, .1, 1)

        # update state
        # s' = s + a + noise
        s1 = s + burst_prob + noise

        original_throughput = float(self._reward_params['B_0']) / s

        if self._print_mode:
            print('state:', s, 'action:', action, 'burst_prob:', burst_prob, 's1:', s1,
                  'original_throughput:', original_throughput)

        reward, burst_prob, ppc_data_mb_scheduled, user_lost_data_mb, new_cell_throughput = \
            get_simplified_reward(action=burst_prob,
                                  reward_params=self._reward_params,
                                  print_mode=self._print_mode,
                                  burst_prob_user_selector='same_as_ppc',
                                  original_throughput=original_throughput,
                                  hard_throughput_limit_flag=self._hard_throughput_limit_flag)
        self._reward_history = report_rewards(state=s,
                                              burst_prob=burst_prob,
                                              reward=reward,
                                              reward_history=self._reward_history,
                                              iteration_index=self._iteration_index,
                                              ppc_data_mb_scheduled=ppc_data_mb_scheduled,
                                              user_lost_data_mb=user_lost_data_mb,
                                              print_mode=self._print_mode,
                                              original_throughput=original_throughput,
                                              new_throughput=new_cell_throughput,
                                              throughput_var=self._throughput_var,
                                              batch_number=self._batch_number)

        # For debugging
        self.reward = reward
        self.action_taken = burst_prob

        self.state = s1
        self._iteration_index += 1
        done = False

        # abort if throughput too low
        if self._iteration_index >= self._min_iterations_before_done:
            # noinspection PyUnresolvedReferences
            n_low_throughput_entries = (self._reward_history[self._throughput_var][-self._n_last_entries:] <=
                                        self._bad_throughput_threshold).sum()
            fraction_low_throughput = n_low_throughput_entries / self._n_last_entries
            if fraction_low_throughput >= .9:
                done = True
                if self._print_mode:
                    print('Abort due to low throughput')
                    print('n_low_throughput_entries:', n_low_throughput_entries,
                          'bad_throughput_threshold:', self._bad_throughput_threshold,
                          'last:', self._reward_history[self._throughput_var][-self._n_last_entries:])

        if self._print_mode:
            print('Step', 'state:', self.state, 's1:', s1, 'reward:', reward, 'done:', done)

        return s1, reward, done, {}

    def reset(self):
        """ Reset state for a new episode """
        if self._deterministic_reset_mode:
            self.state = self._reset_state_vector
        else:
            # biased sampling of C (collision) to lower congestion states for beginning of system
            self.state = self._random.uniform(low=self._min_state_vector, high=(self._max_state_vector / 5))

        self._batch_number += 1
        self._iteration_index = 0

        if self._print_mode:
            print('Reset')
            print('mode:', self._deterministic_reset_mode, 'state:', self.state)

        return self.state

    def render(self, mode='human'):
        pass
