from argparse import ArgumentParser
from common.load_keras_datasets import load_cifar10_dataset
from common.util import merge_dict
from fastai.conv_learner import ConvLearner, resnet34, tfms_from_model
from fastai.dataset import ImageClassifierData
import numpy as np
import os
from pytorch_model.cnn.hyperparams import get_constants


OUTPUT_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/output/')


# noinspection SpellCheckingInspection
def run(constant_overwrites):
    x_train, y_train, x_test, y_test = load_cifar10_dataset()
    # n_classes = 10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    print('')
    print('original:')
    print('X_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(y_train[:3])

    # not required when using `tfms_from_model` below
    # x_train = x_train.reshape(x_train.shape[0], -1)
    x_train = x_train.astype(np.float32)  # WTF? float/float64 (default) raises error
    y_train = np.where(y_train == 1)[1]
    y_train = y_train.astype(np.int)  # uint8 is causing error
    # x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = x_test.astype(np.float32)  # as above
    y_test = np.where(y_test == 1)[1]
    y_test = y_test.astype(np.int)  # as above

    # sample to test on CPU
    x_train = x_train[:800]
    y_train = y_train[:800]
    x_test = x_test[:200]
    y_test = y_test[:200]

    print('')
    print('reshaped:')
    print('X_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(y_train[:3])
    print('X_train dtype:', x_train.dtype)
    print('y_train dtype:', y_train.dtype)

    constants = merge_dict(get_constants(), constant_overwrites)
    constants['img_shape'] = x_train.shape[1:]

    arch = resnet34
    data = ImageClassifierData.from_arrays(OUTPUT_DIR,
                                           trn=(x_train, y_train),
                                           val=(x_test, y_test),
                                           classes=classes,
                                           tfms=tfms_from_model(arch, constants['image_size']))
    learn = ConvLearner.pretrained(arch, data, precompute=True)
    # lrf = learn.lr_find()
    learn.fit(constants['learning_rate'], constants['n_epochs'])

    learn.sched.plot_loss()

    log_preds = learn.predict()
    preds = np.argmax(log_preds, axis=1)

    print('Finished')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Fast.ai CNN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.set_defaults(retrain=False)
    args = parser.parse_args()
    run(vars(args))
