from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import matplotlib.pyplot as plt
import numpy as np
import os
from rl.environments.gridworld import GameEnvironment
from rl.survey_of_methods.deep_recurrent_q_network.util import test, train


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    # Initializing the Gridworld with True limits the field of view, resulting
    # in a partially observable MDP. Initializing it with False provides the
    # agent with the entire environment, resulting in a fully observable MDP.
    env = GameEnvironment(partial=True, size=9)

    n_hidden = constants['n_hidden']
    start_epsilon = constants['start_epsilon']
    end_epsilon = constants['end_epsilon']
    annealing_steps = constants['annealing_steps']
    tau = constants['tau']
    gamma = constants['gamma']
    learning_rate = constants['learning_rate']
    trace_length = constants['trace_length']
    save_path = constants['save_path']
    load_model = constants['load_model']
    n_episodes = constants['n_episodes']
    max_episode_length = constants['max_episode_length']
    n_pretrain_steps = constants['n_pretrain_steps']
    batch_size = constants['batch_size']
    update_freq = constants['update_freq']
    summary_length = constants['summary_length']
    time_per_step = constants['time_per_step']

    rewards, _ = train(env, n_hidden, start_epsilon, end_epsilon, annealing_steps, tau, gamma, learning_rate,
                       trace_length, save_path, load_model, n_episodes, max_episode_length, n_pretrain_steps,
                       batch_size, update_freq, summary_length, time_per_step)

    # Check network learning

    # mean reward over time
    reward_mat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
    mean_reward = np.average(reward_mat, 1)
    plt.plot(mean_reward)
    plt.show()

    print('Test the network...')

    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.test.yml')
    constants = merge_dict(constants, load_hyperparams(config_path))

    n_hidden = constants['n_hidden']
    save_path = constants['save_path']
    load_model = constants['load_model']
    n_episodes = constants['n_episodes']
    max_episode_length = constants['max_episode_length']
    summary_length = constants['summary_length']
    time_per_step = constants['time_per_step']
    epsilon = constants['epsilon']

    rewards, _ = test(env, n_hidden, save_path, load_model, n_episodes, max_episode_length,
                      summary_length, time_per_step, epsilon)

    # mean reward over time
    reward_mat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
    mean_reward = np.average(reward_mat, 1)
    plt.plot(mean_reward)
    plt.show()


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Deep Q-Network')
    parser.add_argument('--episodes', dest='n_episodes', type=int, help='number episodes')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='size of hidden layer')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--load-model', dest='load_model', help='load model flag', action='store_true')
    parser.set_defaults(load_model=False)
    args = parser.parse_args()

    run(vars(args))
