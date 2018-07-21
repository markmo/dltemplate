from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import matplotlib.pyplot as plt
import os
from rl.survey_of_methods.model_based.model_setup import ModelNetwork, PolicyNetwork
from rl.survey_of_methods.model_based.util import train
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    obs_dim = constants['obs_dim']
    policy_n_hidden = constants['policy_n_hidden']
    model_n_hidden = constants['model_n_hidden']
    learning_rate = float(constants['learning_rate'])
    model_batch_size = constants['model_batch_size']
    real_batch_size = constants['real_batch_size']
    max_episodes = constants['max_episodes']

    print('learning_rate:', learning_rate)

    tf.reset_default_graph()

    policy = PolicyNetwork(obs_dim, policy_n_hidden, learning_rate)
    model = ModelNetwork(model_n_hidden, learning_rate)

    env = gym.make('CartPole-v0')

    p_state, next_states_all = train(env, policy, model, model_batch_size, real_batch_size, max_episodes)

    # Check model representation
    # examine how well the model is able to approximate the true environment after training
    plt.figure(figsize=(8, 12))
    for i in range(6):
        plt.subplot(6, 2, 2 * i + 1)
        plt.plot(p_state[:, i])
        plt.subplot(6, 2, 2 * i + 1)
        plt.plot(next_states_all[:, i])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Model-based RL example')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    args = parser.parse_args()

    run(vars(args))
