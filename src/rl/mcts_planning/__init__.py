from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import os
from pickle import loads
from rl.mcts_planning.util import plan_mcts, Root, train
from rl.util import WithSnapshots

# disable warnings
gym.logger.set_level(40)


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    env = WithSnapshots(gym.make('CartPole-v0'))
    root_observation = env.reset()
    root_snapshot = env.get_snapshot()
    n_actions = env.action_space.n
    root = Root(env, n_actions, root_snapshot, root_observation)

    plan_mcts(root, n_iters=constants['n_iters'])

    test_env = loads(root_snapshot)

    train(root, test_env, show=False)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run CartPole MCTS model')
    parser.add_argument('--iters', dest='n_iters', type=int, help='number iterations')
    args = parser.parse_args()

    run(vars(args))
