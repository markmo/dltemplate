from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import os
from rl.reinforce_cartpole.model_setup import PolicyEstimator
from rl.reinforce_cartpole.util import record_sessions, train
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    env = gym.make('CartPole-v0')
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    sess = tf.Session()
    agent = PolicyEstimator(state_dim, n_actions)
    train(env, agent, n_actions, sess, constants['n_epochs'], constants['n_iters'])

    record_sessions(env.spec.id, agent, n_actions, sess)

    env.close()


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run REINFORCE on Taxi environment')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    args = parser.parse_args()

    run(vars(args))
