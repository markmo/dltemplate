from argparse import ArgumentParser
from common.load_data import load_mnist_dataset
from common.util import merge_dict
import matplotlib.pyplot as plt
from tf_model.with_estimator.hyperparams import get_constants
from tf_model.with_estimator.model_setup import model_builder
from tf_model.util import train_using_estimator


def run(constant_overwrites):
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_dataset(flatten=True)

    plt.figure(figsize=[6, 6])
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title('Label: %i' % y_train[i])
        plt.imshow(x_train[i].reshape([28, 28]), cmap='gray')

    plt.pause(1)  # block to show sample image

    constants = merge_dict(get_constants(), constant_overwrites)

    # using estimator does not require labels to be one-hot encoded first
    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_val,
        'y_val': y_val,
        'X_test': x_test,
        'y_test': y_test
    }

    metrics = train_using_estimator(data, model_builder, constants)
    print('')
    print('Test accuracy:', metrics['accuracy'])


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run homemade model')
    parser.add_argument('--epochs', dest='n_epochs', help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', help='number hidden layers')
    args = parser.parse_args()

    run(vars(args))
