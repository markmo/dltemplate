from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import matplotlib.pyplot as plt
import numpy as np
import os
from rl.environments.gridworld import GameEnvironment
from rl.survey_of_methods.dqn.util import train


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    env = GameEnvironment(partial=False, size=5)

    n_hidden = constants['n_hidden']
    start_epsilon = constants['start_epsilon']
    end_epsilon = constants['end_epsilon']
    annealing_steps = constants['annealing_steps']
    tau = constants['tau']
    gamma = constants['gamma']
    save_path = constants['save_path']
    load_model = constants['load_model']
    n_episodes = constants['n_episodes']
    batch_size = constants['batch_size']
    max_episode_length = constants['max_episode_length']
    n_pretrain_steps = constants['n_pretrain_steps']
    update_freq = constants['update_freq']
    learning_rate = constants['learning_rate']

    rewards, _ = train(env, n_hidden, start_epsilon, end_epsilon, annealing_steps, tau, gamma, learning_rate,
                       save_path, load_model, n_episodes, batch_size, max_episode_length, n_pretrain_steps, update_freq)

    # Check network learning

    # mean reward over time
    reward_mat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
    mean_reward = np.average(reward_mat, 1)
    plt.plot(mean_reward)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Deep Q-Network')
    parser.add_argument('--episodes', dest='n_episodes', type=int, help='number episodes')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--load-model', dest='load_model', help='load model flag', action='store_true')
    parser.set_defaults(load_model=False)
    args = parser.parse_args()

    run(vars(args))
