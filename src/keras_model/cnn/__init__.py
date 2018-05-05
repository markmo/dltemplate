from argparse import ArgumentParser
from common.load_data import load_cifar10_dataset
from common.util import merge_dict
from common.util_keras import ModelSaveCallback, reset_tf_session, TqdmProgressCallback
from keras_model.cnn.hyperparams import get_constants
from keras_model.cnn.model_setup import network_builder, model_builder
from keras import backend as ke
from keras.callbacks import Callback, LearningRateScheduler
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os


def run(constant_overwrites):
    x_train, y_train, x_test, y_test = load_cifar10_dataset()
    print('X shape:', x_train.shape)
    n_classes = 10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # show random images from training set
    cols = 8
    rows = 2
    fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            idx = np.random.randint(0, len(y_train))
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            ax.imshow(x_train[idx, :])
            ax.set_title(classes[y_train[idx, 0]])

    plt.show()

    constants = merge_dict(get_constants(), constant_overwrites)
    constants['img_shape'] = x_train.shape[1:]
    constants['n_classes'] = n_classes
    model_filename = constants['model_filename']
    reset_tf_session()
    model = model_builder(network_builder, constants)
    if os.path.exists(model_filename) and not constants['retrain']:
        model = load_model(model_filename.format(constants['n_epochs']))
    else:
        initial_learning_rate = constants['learning_rate']

        # scheduler of learning rate (decay with epochs)
        def lr_scheduler(epoch):
            return initial_learning_rate * 0.9 ** epoch

        # callback for printing of actual learning rate used by optimizer
        class LrHistory(Callback):
            def on_epoch_begin(self, epoch, logs=None):
                # if logs is None:
                #     logs = {}
                print('Learning rate:', ke.get_value(model.optimizer.lr))

        last_finished_epoch = None

        model.fit(x_train, y_train,
                  batch_size=constants['batch_size'],
                  epochs=constants['n_epochs'],
                  callbacks=[LearningRateScheduler(lr_scheduler),
                             LrHistory(),
                             TqdmProgressCallback(),
                             ModelSaveCallback(model_filename)],
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=0,
                  initial_epoch=last_finished_epoch or 0)

        model.save_weights('weights.h5')


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run Keras CNN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.set_defaults(retrain=False)
    args = parser.parse_args()
    run(vars(args))
