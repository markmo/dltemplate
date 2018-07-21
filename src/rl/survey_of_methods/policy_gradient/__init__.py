from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import os
from rl.survey_of_methods.policy_gradient.model_setup import Agent
from rl.survey_of_methods.policy_gradient.util import train
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    env = gym.make('CartPole-v0')

    # hyperparams
    learning_rate = constants['learning_rate']
    n_hidden = constants['n_hidden']
    gamma = constants['gamma']
    n_epochs = constants['n_epochs']

    agent = Agent(learning_rate=learning_rate, state_dim=4, n_actions=2, n_hidden=n_hidden)

    tf.reset_default_graph()
    sess = tf.Session()

    train(env, agent, sess, gamma=gamma, n_epochs=n_epochs)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Policy Gradient RL model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='hidden layer size')
    args = parser.parse_args()

    run(vars(args))
