import numpy as np
import pickle


def discount_rewards(rewards, gamma):
    """ Take 1D float array of rewards and compute discounted rewards """
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0  # reset the sum since this was a game boundary (Pong specific!)

        running_add *= gamma + rewards[t]
        discounted_rewards[t] = running_add

    return discounted_rewards


def load_model(path):
    return pickle.load(open(path, 'rb'))


def log_episode(reward_sum, running_reward):
    print('Resetting env episode reward total was %f. running mean: %f' % (reward_sum, running_reward))


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x


def preprocess(x):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    x = x[35:195]       # crop
    x = x[::2, ::2, 0]  # downsample by factor of 2
    x[x == 144] = 0     # erase background (background type 1)
    x[x == 109] = 0     # erase background (background type 2)
    x[x != 0] = 1       # everything else (paddles, ball) just set to 1
    return x.astype(np.float).ravel()


def save_model(model, path):
    pickle.dump(model, open(path, 'wb'))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # sigmoid "squashing" function to interval [0, 1]
