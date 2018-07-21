from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import numpy as np
import os
from rl.survey_of_methods.multi_armed_bandit.model_setup import Agent
from rl.survey_of_methods.multi_armed_bandit.util import train
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    bandits = [0.2, 0, -0.2, -5]
    n_bandits = len(bandits)
    learning_rate = constants['learning_rate']
    n_episodes = constants['n_episodes']
    epsilon = constants['epsilon']

    tf.reset_default_graph()
    sess = tf.Session()

    agent = Agent(n_bandits, learning_rate)

    _, w1 = train(bandits, agent, sess, n_episodes=n_episodes, epsilon=epsilon)

    print('The agent thinks bandit %s is the most promising' % str(np.argmax(w1) + 1))
    if np.argmax(w1) == np.argmax(-np.array(bandits)):
        print('and it is right')
    else:
        print('and it is wrong')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Q-Table Learning model')
    parser.add_argument('--episodes', dest='n_episodes', type=int, help='number episodes')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--epsilon', dest='epsilon', type=float, help='chance of taking random action')
    args = parser.parse_args()

    run(vars(args))
