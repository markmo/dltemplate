from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


def get_regret(env, agents, n_steps=5000, n_trials=50):
    scores = OrderedDict({agent.name: [0.0 for _ in range(n_steps)] for agent in agents})
    for _ in trange(n_trials):
        env.reset()
        for a in agents:
            a.init_actions(env.action_count)

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()
            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - reward

            env.step()  # change bandit's state if it is non-stationary

    for agent in agents:
        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials

    return scores


def plot_regret(scores, agents):
    for agent in agents:
        plt.plot(scores[agent.name])

    plt.legend([agent.name for agent in agents])
    plt.ylabel('regret')
    plt.xlabel('steps')
    plt.show()
