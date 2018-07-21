import numpy as np


def train(env, q_table, n_episodes=2000, learning_rate=0.8, gamma=0.95):
    rewards = []
    for i in range(n_episodes):
        s = env.reset()  # reset environment and get first new observation
        total_reward = 0
        j = 0
        while j < 99:
            j += 1
            # Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(q_table[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

            s1, reward, done, _ = env.step(a)

            # update table
            q_table[s, a] = q_table[s, a] + learning_rate * (reward + gamma * np.max(q_table[s1, :]) - q_table[s, a])
            total_reward += reward
            s = s1
            if done:
                break

        rewards.append(total_reward)

    return rewards
