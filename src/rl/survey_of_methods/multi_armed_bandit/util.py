import numpy as np
import tensorflow as tf


def pull_bandit(bandit):
    """
    Generates a random number from a normal distribution with a mean of 0. The lower
    the bandit number, the more likely a positive reward will be returned. We want
    our agent to learn to always choose the bandit that will give that positive reward.

    :param bandit:
    :return:
    """
    result = np.random.randn(1)
    return 1 if result > bandit else -1


def train(bandits, agent, sess, n_episodes=1000, epsilon=0.1):
    """
    Train our agent by taking actions in our environment, and receiving rewards.
    Using the rewards and actions, we can know how to properly update our network
    in order to more often choose actions that will yield the highest rewards over time.

    :param bandits:
    :param agent:
    :param sess:
    :param n_episodes:
    :param epsilon:
    :return: tuple of total reward and bandit weights
    """
    n_bandits = len(bandits)
    total_reward = np.zeros(n_bandits)
    sess.run(tf.global_variables_initializer())
    w1 = np.zeros_like(bandits)
    for i in range(n_episodes):
        # explore else exploit
        if np.random.rand(1) < epsilon:
            action = np.random.randint(n_bandits)
        else:
            action = sess.run(agent.a)

        reward = pull_bandit(bandits[action])

        # Update the network
        _, resp_w, w1 = sess.run([agent.update_op, agent.responsible_weight, agent.w], feed_dict={
            agent.reward_ph: [reward],
            agent.action_ph: [action]
        })

        # Update the running tally of scores
        total_reward[action] += reward

        if i % 50 == 0:
            print('Running reward for the %i bandits: %s' % (n_bandits, str(total_reward)))

    return total_reward, w1
