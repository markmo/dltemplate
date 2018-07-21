import numpy as np
import tensorflow as tf


def train(bandit, agent, w, sess, n_episodes=10000, epsilon=0.1):
    sess.run(tf.global_variables_initializer())
    total_reward = np.zeros([bandit.n_bandits, bandit.n_actions])  # set scoreboard for bandits to zeros
    w1 = np.zeros([bandit.n_bandits, bandit.n_actions])
    for i in range(n_episodes):
        s = bandit.get_bandit()  # get a state from the environment

        # explore else exploit
        if np.random.rand(1) < epsilon:
            action = np.random.randint(bandit.n_actions)
        else:
            action = sess.run(agent.chosen_action, feed_dict={agent.state_in: [s]})

        reward = bandit.pull_arm(action)  # get our reward for taking an action given a bandit

        # Update the network
        feed_dict = {agent.reward_ph: [reward], agent.action_ph: [action], agent.state_in: [s]}
        _, w1 = sess.run([agent.update_op, w], feed_dict)

        # Update the running tally of rewards
        total_reward[s, action] += reward

        if i % 500 == 0:
            print('Mean reward for each of the %i bandits: %s' %
                  (bandit.n_bandits, str(np.mean(total_reward, axis=1))))

    return total_reward, w1
