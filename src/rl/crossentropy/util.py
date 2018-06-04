import gym
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import numpy as np


def generate_session(env, policy, n_actions, t_max=10**4):
    """
    Play game until done, or for t_max clicks.

    :param env: OpenAI Gym environment
    :param policy: an array of shape [n_states, n_actions] with action probabilities
    :param n_actions:
    :param t_max:
    :return: list of states, list of actions, and sum of rewards
    """
    states, actions = [], []
    total_reward = 0
    s = env.reset()
    for t in range(t_max):
        a = np.random.choice(n_actions, p=policy[s])
        new_s, r, done, info = env.step(a)

        # Record state, action and add up reward to states, actions,
        # and total_reward accordingly
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    return states, actions, total_reward


def generate_agent_session(env, agent, n_actions, t_max=10**4):
    """
    Play game until done, or for t_max clicks.

    :param env: OpenAI Gym environment
    :param agent: ANN classifier
    :param n_actions:
    :param t_max:
    :return: list of states, list of actions, and sum of rewards
    """
    states, actions = [], []
    total_reward = 0
    s = env.reset()
    for t in range(t_max):
        # predict array of action probabilities
        probs = agent.predict_proba([s])[0]
        a = np.random.choice(n_actions, p=probs)
        new_s, r, done, info = env.step(a)

        # Record state, action and add up reward to states, actions,
        # and total_reward accordingly
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    return states, actions, total_reward


def record_sessions(env_id, agent, n_actions):
    env = Monitor(gym.make(env_id), directory='videos', force=True)
    for _ in range(100):
        generate_agent_session(env, agent, n_actions)

    env.close()


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile

    Make sure to return elite states and actions in their original order,
    i.e. sorted by session number and timestep within session.

    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]
    :param percentile:
    :return: (elite_states, elite_actions), both 1D lists of states and
             respective actions from elite sessions
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    mask = rewards_batch > reward_threshold
    if mask.any():
        elite_states = np.concatenate(np.array(states_batch)[mask]).tolist()
        elite_actions = np.concatenate(np.array(actions_batch)[mask]).tolist()
    else:
        elite_states, elite_actions = [], []

    return elite_states, elite_actions


def show_progress(rewards_batch, log, percentile, reward_range=None):
    """
    Convenience function that displays training progress.

    :param rewards_batch:
    :param log:
    :param percentile:
    :param reward_range:
    :return:
    """
    if reward_range is None:
        reward_range = [-990, 10]

    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    print('mean reward = %.3f, threshold = %.3f' % (mean_reward[0], threshold))

    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([np.percentile(rewards_batch, percentile)], [0], [100], label='percentile', color='red')
    plt.legend()
    plt.grid()
    plt.show()


def update_policy(elite_states, elite_actions, n_states, n_actions):
    """
    Given old policy and a list of elite states, actions from `select_elites`,
    return new updated policy where each action probability is proportional to

    policy[s_i, a_i] ~ #[occurrences of si and ai in elite states, actions]

    Don't forget to normalize policy to get valid probabilities, and to handle
    0, 0 case. In case you never visited a state, set probabilities for all
    actions to 1./n_actions.

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions
    :param n_states:
    :param n_actions:
    :return:
    """
    new_policy = np.zeros((n_states, n_actions))
    for s, a in zip(elite_states, elite_actions):
        new_policy[s, a] += 1

    # normalize
    for s in range(n_states):
        tot = np.sum(new_policy[s])
        if tot == 0:
            new_policy[s, :] = 1. / n_actions
        else:
            new_policy[s] /= tot

    return new_policy
