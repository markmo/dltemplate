from collections import deque
import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf


def generate_session(env, agent, n_actions, sess, t_max=1000):
    """ play env with REINFORCE agent and train at the session end """
    states, actions, rewards = [], [], []
    s = env.reset()
    for t in range(t_max):
        # action probabilities (pi(a|s))
        action_probs = agent.get_action_probs(s, sess)
        a = np.random.choice(n_actions, 1, p=action_probs)[0]
        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    train_step(agent, states, actions, rewards, sess)
    return sum(rewards)


def get_cumulative_rewards(rewards, gamma=0.99):
    """
    Take a list of immediate rewards r(s,a) for the whole session.

    Compute cumulative rewards R(s,a) (aka. G(s,a) in Sutton '16)
    R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last
    to first time tick and compute R_t = r_t + gamma*R_{t+1} recurrently

    :param rewards: rewards at each step
    :param gamma: discount for reward
    :return: (list) of cumulative rewards with as many elements as in the initial rewards
    """
    cum_rewards = deque([rewards[-1]])
    for i in range(len(rewards)-2, -1, -1):
        cum_rewards.appendleft(rewards[i] + gamma * cum_rewards[0])

    return cum_rewards


def record_sessions(env_id, agent, n_actions, sess):
    env = Monitor(gym.make(env_id), directory='videos', force=True)
    for _ in range(100):
        generate_session(env, agent, n_actions, sess)

    env.close()


def train(env, agent, n_actions, sess=None, n_epochs=100, n_iter=100, mean_reward_threshold=300):
    if not sess:
        sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs):
        rewards = [generate_session(env, agent, n_actions, sess) for _ in range(n_iter)]
        print('mean reward: %.3f' % np.mean(rewards))
        if np.mean(rewards) > mean_reward_threshold:
            break


def train_step(agent, states, actions, rewards, sess):
    """ given full session, trains agent with policy gradient """
    cum_rewards = get_cumulative_rewards(rewards)
    agent.update(states, actions, cum_rewards, sess)
