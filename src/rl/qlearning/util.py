from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def play_and_train(env, agent, t_max=10**4):
    """
    This function should
    1. run a full game, actions given by agent's e-greedy policy
    2. train agent using agent.update(...) whenever it is possible
    3. return total reward

    :param env:
    :param agent:
    :param t_max:
    :return:
    """
    total_reward = 0.
    s = env.reset()
    for t in range(t_max):
        # get agent to pick action given state s
        a = agent.get_action(s)
        next_s, r, done, _ = env.step(a)
        agent.update(s, a, r, next_s)
        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


def train(env, agent, n_epochs=1000):
    rewards = []
    for i in range(n_epochs):
        rewards.append(play_and_train(env, agent))
        agent.epsilon *= 0.99
        if i % 100 == 0:
            clear_output(True)
            print('eps = %.5f' % agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
            plt.plot(rewards)
            plt.show()
