from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import numpy as np
import os
from rl.survey_of_methods.contextual_bandits.model_setup import ContextualBandit, Agent
from rl.survey_of_methods.contextual_bandits.util import train
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    learning_rate = constants['learning_rate']
    n_episodes = constants['n_episodes']
    epsilon = constants['epsilon']

    tf.reset_default_graph()

    bandit = ContextualBandit()
    agent = Agent(learning_rate=learning_rate, state_dim=bandit.n_bandits, n_actions=bandit.n_actions)
    w = tf.trainable_variables()[0]

    sess = tf.Session()

    _, w1 = train(bandit, agent, w, sess, n_episodes=n_episodes, epsilon=epsilon)

    for i in range(bandit.n_bandits):
        print('The agent thinks action %s for bandit %i is the most promising' %
              (str(np.argmax(w1[i]) + 1), i + 1))

        if np.argmax(w1[i]) == np.argmin(bandit.bandits[i]):
            print('and it is right!')
        else:
            print('and it is wrong!')

        print('')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Contextual Bandit')
    parser.add_argument('--episodes', dest='n_episodes', type=int, help='number episodes')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--epsilon', dest='epsilon', type=float, help='chance of taking random action')
    args = parser.parse_args()

    run(vars(args))
