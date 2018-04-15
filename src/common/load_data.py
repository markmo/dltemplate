import keras


def load_mnist_dataset(flatten=False):
    # loads into ~/.keras/datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # normalize X
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # we reserve the last 10000 training examples for validation
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        x_train = x_train.reshape([x_train.shape[0], -1])
        x_val = x_val.reshape([x_val.shape[0], -1])
        x_test = x_test.reshape([x_test.shape[0], -1])

    return x_train, y_train, x_val, y_val, x_test, y_test
