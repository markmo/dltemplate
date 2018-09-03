import keras


def load_cifar10_dataset():
    """
    The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes,
    with 6,000 images per class. There are 50,000 training images and 10,000
    test images.

    The dataset is divided into five training batches and one test batch,
    each with 10,000 images. The test batch contains exactly 1,000
    randomly-selected images from each class. The training batches contain
    the remaining images in random order, but some training batches may contain
    more images from one class than another. Between them, the training batches
    contain exactly 5,000 images from each class.

    The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse,
    ship, and truck.

    The classes are completely mutually exclusive. There is no overlap between
    automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort.
    "Truck" includes only big trucks. Neither includes pickup trucks.

    :return:
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train / 255. - 0.5
    x_test = x_test / 255. - 0.5

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def load_mnist_dataset(flatten=False):
    """
    MNIST database of handwritten digits.

    Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

    :param flatten: boolean setting to flatten pixel matrix to vector
    :return: dataset divided into features and labels for training, validation and test
    """
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
