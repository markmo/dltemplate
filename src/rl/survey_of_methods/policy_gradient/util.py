import numpy as np
from rl.util import discount_rewards
import tensorflow as tf


def train(env, agent, sess, gamma=0.99, n_epochs=5000, max_episodes=999, update_freq=5):
    sess.run(tf.global_variables_initializer())
    total_reward = []
    total_length = []
    grad_buffer = np.array(sess.run(tf.trainable_variables()))
    grad_buffer *= 0

    for i in range(n_epochs):
        s = env.reset()
        running_reward = 0
        states, actions, rewards = [], [], []
        for j in range(max_episodes):
            # probabilistically pick an action given our network outputs
            a_dist = sess.run(agent.output, feed_dict={agent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1, reward, done, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(reward)
            s = s1
            running_reward += reward

            if done:
                # episode lengths vary

                # Update the network
                discounted_rewards = discount_rewards(rewards, gamma)
                feed_dict = {
                    agent.reward_ph: discounted_rewards,
                    agent.action_ph: actions,
                    agent.state_in: states
                }
                grads = sess.run(agent.gradients, feed_dict)
                for idx, grad in enumerate(grads):
                    grad_buffer[idx] += grad

                if i % update_freq == 0 and i != 0:
                    feed_dict = dict(zip(agent.gradient_phs, grad_buffer))
                    sess.run(agent.update_batch, feed_dict)
                    grad_buffer *= 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

        # Update the running tally of rewards
        if i % 100 == 0:
            print('Mean reward:', np.mean(total_reward[-100:]))
