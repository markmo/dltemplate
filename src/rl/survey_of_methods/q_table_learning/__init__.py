from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import numpy as np
import os
from rl.survey_of_methods.q_table_learning.util import train


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    env = gym.make('FrozenLake-v0')

    # initialize table with all zeros
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    learning_rate = constants['learning_rate']
    gamma = constants['gamma']
    n_episodes = constants['n_episodes']

    rewards = train(env, q_table, n_episodes, learning_rate, gamma)

    print('Score over time:', str(sum(rewards) / n_episodes))
    print('Final Q-Table values:')
    print(q_table)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Q-Table Learning model')
    parser.add_argument('--episodes', dest='n_episodes', type=int, help='number episodes')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--gamma', dest='gamma', type=float, help='gamma discount factor')
    args = parser.parse_args()

    run(vars(args))
