import matplotlib.pyplot as plt
import numpy as np


# Common utility functions

def merge_dict(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if value is not None:
            merged[key] = value

    return merged


def next_batch(x, y, batch_size):
    """
    Extract mini batches from input and matching labels

    :param x data samples
    :param y data labels
    :param batch_size size of batch
    """
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    x_shuffle = [x[i] for i in idx]
    y_shuffle = [y[i] for i in idx]
    return np.asarray(x_shuffle), np.asarray(y_shuffle)


def one_hot_encode(labels, n_classes):
    n_samples = labels.shape[0]
    encoded = np.zeros((n_samples, n_classes))
    encoded[np.arange(n_samples), labels] = 1
    return encoded


def plot_accuracy(n_epochs, train_costs, val_costs):
    iterations = list(range(n_epochs))
    plt.figure()
    plt.plot(iterations, train_costs, label='Train')
    plt.plot(iterations, val_costs, label='Valid')
    plt.legend()
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()


def reshape(x, n, dtype='float32'):
    return x.reshape(x.shape[0], n).astype(dtype)
