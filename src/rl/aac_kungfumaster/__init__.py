from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import os
from rl.aac_kungfumaster.model_setup import Agent
from rl.aac_kungfumaster.util import evaluate, make_env
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    env = make_env()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print('\nPrint game info:')
    print('observation shape:', obs_shape)
    print('n_actions:', n_actions)
    print('action names:', env.env.env.get_action_meanings())

    # Print game images:
    s = env.reset()
    for _ in range(100):
        s, _, _, _ = env.step(env.action_space.sample())

    plt.title('Game image')
    plt.imshow(env.render('rgb_array'))
    plt.show()

    plt.title('Agent observation (4-frame buffer')
    plt.imshow(s.transpose([0, 2, 1]).reshape([42, -1]))
    plt.show()

    tf.reset_default_graph()
    sess = tf.Session()
    agent = Agent('agent', obs_shape, n_actions)
    sess.run(tf.global_variables_initializer())

    env_monitor = Monitor(env, directory='videos', force=True)
    game_rewards = evaluate(agent, env, sess, n_games=constants['n_sessions'])
    env_monitor.close()
    print('Game rewards:', game_rewards)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Kung Fu Master AAC model')
    parser.add_argument('--sessions', dest='n_sessions', type=int, help='number sessions')
    args = parser.parse_args()

    run(vars(args))
