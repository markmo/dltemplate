from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
from rl.crossentropy.model_setup import agent_builder
from rl.crossentropy.util import generate_agent_session, generate_session, select_elites
from rl.crossentropy.util import show_progress, update_policy


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    n_sessions = constants['n_sessions']  # sample this many sessions
    percentile = constants['percentile']  # take this percent of sessions with highest rewards
    env_id = constants['env_id']

    if env_id == 'Taxi-v2':
        env = gym.make('Taxi-v2')
        env.reset()
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        print('n_states = %i, n_actions = %i' % (n_states, n_actions))

        policy = np.ones((n_states, n_actions)) / n_actions

        # plot initial reward distribution
        sample_rewards = [generate_session(env, policy, n_actions, t_max=1000)[-1] for _ in range(200)]

        plt.hist(sample_rewards, bins=20)
        plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label="50'th percentile", color='green')
        plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label="90'th percentile", color='red')
        plt.legend()
        plt.show()

        learning_rate = constants['learning_rate']
        log = []
        for i in range(100):
            sessions = [generate_session(env, policy, n_actions) for _ in range(n_sessions)]
            states_batch, actions_batch, rewards_batch = zip(*sessions)
            elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
            new_policy = update_policy(elite_states, elite_actions, n_states, n_actions)
            policy = learning_rate * new_policy + (1 - learning_rate) * policy
            show_progress(rewards_batch, log, percentile)

    elif env_id in ('CartPole-v0', 'MountainCar-v0'):
        env = gym.make(env_id).env
        env.reset()
        n_actions = env.action_space.n
        agent = agent_builder(constants)

        # initialize agent to the dimension of state and number of actions
        agent.fit([env.reset()] * n_actions, range(n_actions))

        log = []
        for i in range(100):
            sessions = [generate_agent_session(env, agent, n_actions) for _ in range(n_sessions)]
            states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))
            elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
            agent.fit(elite_states, elite_actions)
            show_progress(rewards_batch, log, percentile)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Crossentropy RL model')
    parser.add_argument('--env', dest='env_id', help='gym environment')
    parser.add_argument('--sessions', dest='n_sessions', type=int, help='number sessions')
    parser.add_argument('--percentile', dest='percentile', type=int, help='percentile')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', type=int, help='number hidden layers')
    args = parser.parse_args()

    run(vars(args))
