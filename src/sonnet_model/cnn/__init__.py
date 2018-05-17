from argparse import ArgumentParser
from common.load_data import load_signs_dataset
from common.util import convert_to_one_hot
import matplotlib.pyplot as plt
import numpy as np
from sonnet_model.cnn.util import train


def run(constants):
    x_train, y_train, x_test, y_test, _ = load_signs_dataset()

    x_train = x_train / 255
    x_test = x_test / 255
    n_classes = 6
    y_train = convert_to_one_hot(y_train, n_classes).T
    y_test = convert_to_one_hot(y_test, n_classes).T

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    # Show image example
    sample_idx = 6
    print('Image example:')
    print('y =', np.argmax(y_train[sample_idx]))
    plt.imshow(x_train[sample_idx])
    plt.show()

    learning_rate = constants['learning_rate']
    n_epochs = constants['n_epochs']
    train(x_train, y_train, x_test, y_test, learning_rate=learning_rate, n_epochs=n_epochs)


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run Keras CNN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    args = parser.parse_args()
    run(vars(args))
