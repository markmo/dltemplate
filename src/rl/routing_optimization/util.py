""" util """
__credits__ = 'giorgio@ac.upc.edu'

from collections import deque, OrderedDict
import json
import numpy as np
import os
import time

ROOT = os.path.dirname(os.path.realpath(__file__)) + '/'


def format_float(f):
    try:
        float(f)
        return str.format('{0:.3f}', f).rstrip('0').rstrip('.')
    except ValueError:
        return str(f)


def load_experiment(constants):
    experiment_filename = constants['experiment']
    with open(experiment_filename) as f:
        config = json.load(f)

    cluster = constants['cluster']
    expt = experiment_filename.lower().split('.')[0]
    config['cluster'] = cluster
    if cluster == 'local':
        # TODO
        # import experiment.local
        # run_wrapper = experiment.local.run_wrapper
        run_wrapper = None
        config['experiment'] = setup_experiment(expt)
    else:
        run_wrapper = None
        config['experiment'] = expt

    return config, run_wrapper


def scale(array):
    mean = array.mean()
    std = array.std()
    if std == 0:
        std = 1

    return np.asarray((array - mean) / std)


# noinspection SpellCheckingInspection
def selu(x):
    """
    Scaled Exponential Linear Unit. (Klambauer et al., 2017)

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

    :param x: tensor or variable to compute the activation function
    :return:
    """
    from keras.activations import elu
    alpha = 1.6732632423543772848170429916717
    scale_ = 1.0507009873554804934193349852946

    return scale_ * elu(x, alpha)


def setup(folder, constants):
    os.makedirs(folder, exist_ok=True)
    with open(folder + 'folder.ini', 'w') as f:
        f.write('[General]\n')
        f.write('**.folderName = "{}"\n'.format(folder))

    with open(folder + 'hyperparams.json', 'w') as f:
        json.dump(OrderedDict(sorted(constants.items(), key=lambda t: t[0])), f, indent=4)

    if constants['traffic'].startswith('stat:'):
        with open(folder + 'Traffic.txt', 'w') as f:
            f.write(constants['traffic'].split('stat:')[-1] + '\n')

    return folder


def setup_brute(constants):
    epoch = 't%.6f/' % time.time()
    folder = ROOT + 'runs/brute' + epoch.replace('.', '') + '/'
    return setup(folder, constants)


def setup_experiment(experiment=''):
    folder = ROOT + 'runs/'
    os.makedirs(folder, exist_ok=True)
    folder += experiment + '/'
    os.makedirs(folder, exist_ok=True)
    return folder


def setup_run(constants):
    epoch = 't%.6f/' % time.time()
    folder = constants['experiment'] + epoch.replace('.', '')
    return setup(folder, constants)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.n_episodes = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # randomly sample batch_size examples
        if self.n_episodes < batch_size:
            indices = np.random.choice(len(self.buffer), self.n_episodes)
        else:
            indices = np.random.choice(len(self.buffer), batch_size)

        return np.asarray(self.buffer)[indices]

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        episode = (state, action, reward, new_state, done)
        if self.n_episodes < self.buffer_size:
            self.buffer.append(episode)
            self.n_episodes += 1
        else:
            self.buffer.popleft()  # TODO - redundant?
            self.buffer.append(episode)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return episode counter
        return self.n_episodes

    def erase(self):
        self.buffer = deque()
        self.n_episodes = 0
