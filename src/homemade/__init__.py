from argparse import ArgumentParser
from common.load_data import load_mnist_dataset
from common.util import merge_dict
from homemade.model_setup import network_builder
from homemade.util import forward, iterate_minibatches, predict, train
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def get_constants():
    return {
        'n_classes': 10,
        'n_hidden1': 100,
        'n_hidden2': 200,
        'keep_prob': 1,
        'n_epochs': 5,
        'batch_size': 32
    }


def run(constant_overwrites):
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_dataset(flatten=True)

    plt.figure(figsize=[6, 6])
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title('Label: %i' % y_train[i])
        plt.imshow(x_train[i].reshape([28, 28]), cmap='gray')

    constants = merge_dict(get_constants(), constant_overwrites)
    network = network_builder(x_train, constants)

    train_log = []
    val_log = []
    for epoch in range(constants['n_epochs']):
        for x_batch, y_batch in iterate_minibatches(x_train, y_train,
                                                    batch_size=constants['batch_size'],
                                                    shuffle=True):
            train(network, x_batch, y_batch)

        train_log.append(np.mean(predict(network, x_train) == y_train))
        val_log.append(np.mean(predict(network, x_val) == y_val))

        clear_output()
        print('Epoch', epoch)
        print('Train accuracy:', train_log[-1])
        print('Val accuracy:', val_log[-1])
        if len(train_log) > 1:
            plt.figure()
            plt.plot(train_log, label='train accuracy')
            plt.plot(val_log, label='val accuracy')
            plt.legend(loc='best')
            plt.grid()
            plt.show()


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run homemade model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    args = parser.parse_args()

    run(vars(args))
