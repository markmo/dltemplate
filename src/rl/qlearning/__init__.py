from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import os
from rl.qlearning.qlearning_agent import QLearningAgent
from rl.qlearning.util import train


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    env = gym.make('Taxi-v2')
    n_actions = env.action_space.n
    agent = QLearningAgent(alpha=constants['alpha'],
                           epsilon=constants['epsilon'],
                           discount=constants['discount'],
                           get_legal_actions=lambda s: range(n_actions))

    train(env, agent, constants['n_epochs'])


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Q-Learning on Taxi environment')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='alpha', type=float, help='learning rate')
    parser.add_argument('--epsilon', dest='epsilon', type=float, help='exploration probability')
    parser.add_argument('--discount', dest='discount', type=float, help='discount rate')
    args = parser.parse_args()

    run(vars(args))
