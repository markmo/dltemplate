from argparse import ArgumentParser
from common.load_data import load_mnist_dataset
from common.util import merge_dict, one_hot_encode
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_model.simple.hyperparams import get_constants
from tf_model.simple.model_setup import get_parameters, model_builder, network_builder
from tf_model.setup import get_inputs
from tf_model.util import train


def run(constant_overwrites):
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_dataset(flatten=True)

    plt.figure(figsize=[6, 6])
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title('Label: %i' % y_train[i])
        plt.imshow(x_train[i].reshape([28, 28]), cmap='gray')

    constants = merge_dict(get_constants(), constant_overwrites)
    n_classes = constants['n_classes']
    data = {
        'X_train': x_train,
        'y_train': one_hot_encode(y_train, n_classes),
        'X_val': x_val,
        'y_val': one_hot_encode(y_val, n_classes),
        'X_test': x_test,
        'y_test': one_hot_encode(y_test, n_classes)
    }
    input_x, input_y = get_inputs(constants)
    parameters = get_parameters(constants)
    optimizer, loss_op, predict_op, model, y_ = model_builder(network_builder, input_x, input_y, parameters, constants)
    saver = tf.train.Saver()
    # tf.add_to_collection('predict_op', predict_op)
    train(data, constants, (input_x, input_y), optimizer, loss_op, model, y_, minibatch=True, saver=saver)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run TF Simple NN model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', type=int, help='number hidden layers')
    args = parser.parse_args()

    run(vars(args))
