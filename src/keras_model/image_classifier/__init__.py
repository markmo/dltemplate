from argparse import ArgumentParser
from common.load_data import load_flowers
from common.util import decode_image_from_raw_bytes, merge_dict, prepare_raw_bytes_for_model, read_raw_from_tar
from common.util_keras import reset_tf_session
from keras.models import load_model
from keras_model.image_classifier.hyperparams import get_constants
from keras_model.image_classifier.model_setup import model_builder
from keras_model.image_classifier.util import compile_model, train, train_generator
import matplotlib.pyplot as plt
import os


# Training data
# Takes 12 min and 400 MB.
# * http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
# * http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
# * http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat


def run(constant_overwrites):
    constants = merge_dict(get_constants(), constant_overwrites)
    tar_filename = constants['tar_filename']
    img_size = constants['img_size']

    train_files, test_files, train_labels, test_labels, n_classes = \
        load_flowers(os.path.dirname(os.path.abspath(__file__)))
    data = {
        'train_files': train_files,
        'train_labels': train_labels,
        'test_files': test_files,
        'test_labels': test_labels,
        'n_classes': n_classes
    }

    # test cropping
    raw_bytes = read_raw_from_tar(tar_filename, 'jpg/image_00001.jpg')
    img = decode_image_from_raw_bytes(raw_bytes)

    print('')
    print('original image shape:', img.shape)
    print('')
    plt.imshow(img)
    plt.show()

    img = prepare_raw_bytes_for_model(raw_bytes, img_size, normalize_for_model=False)
    print('')
    print('cropped image shape:', img.shape)
    print('')
    plt.imshow(img)
    plt.show()

    # remember to clear session if you start building graph from scratch!
    # don't call K.set_learning_phase() !!! (otherwise will enable dropout
    # in train/test simultaneously)
    _ = reset_tf_session()  # returns session

    model = model_builder(n_classes, constants)

    print('')
    print(model.summary())
    print('')

    compile_model(model, constants)

    # model_file_exists = any(f.startswith('flowers') for f in os.listdir('.') if os.path.isfile(f))
    last_finished_epoch = constants['last_finished_epoch']
    if last_finished_epoch:
        model = load_model(constants['model_filename'].format(last_finished_epoch))

    train(model, data, constants)

    # Accuracy on test set
    test_accuracy = model.evaluate_generator(
        train_generator(tar_filename, test_files, test_labels, n_classes, constants),
        len(test_files) // constants['batch_size'] // 2
    )[1]

    print('\nTest accuracy: %.5f' % test_accuracy)


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run Keras Image Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--last-finished-epoch', dest='last_finished_epoch', type=int,
                        help='number of last finished epoch')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--tar-filename', dest='tar_filename', help='data tar filename')
    parser.add_argument('--imagenet', dest='use_imagenet', help='use imagenet flag', action='store_true')
    parser.set_defaults(use_imagenet=True)
    args = parser.parse_args()

    run(vars(args))
