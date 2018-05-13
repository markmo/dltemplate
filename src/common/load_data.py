from ast import literal_eval
from common import util_download
from common.util import get_all_filenames
import cv2
import keras
import nltk
import numpy as np
import os
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import tarfile
import tqdm
from urllib.parse import quote as url_quote


DATA_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/data/')

READONLY_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/readonly/')

# http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
ATTRS_NAME = DATA_DIR + 'lfw/lfw_attributes.txt'

# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
# noinspection SpellCheckingInspection
IMAGES_NAME = DATA_DIR + 'lfw/lfw-deepfunneled.tgz'

# http://vis-www.cs.umass.edu/lfw/lfw.tgz
RAW_IMAGES_NAME = DATA_DIR + 'lfw/lfw.tgz'


# noinspection PyUnresolvedReferences
def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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


def load_faces_dataset():
    x, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
    img_shape = x.shape[1:]

    # center images
    x = x.astype('float32') / 255. - 0.5

    # split
    x_train, x_test = train_test_split(x, test_size=0.1, random_state=42)
    return img_shape, attr, x_train, x_test


# noinspection SpellCheckingInspection
def load_flowers(target_path):
    """
    Flowers classification dataset consists of 102 flower categories commonly occurring
    in the United Kingdom. Each class contains between 40 and 258 images.

    http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

    :return:
    """
    util_download.link_all_keras_resources()
    if not os.path.exists(READONLY_DIR + 'keras/models/'):
        # original:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
        util_download.sequential_downloader(
            'https://github.com/hse-aml/intro-to-dl',
            'v0.2',
            [
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
            ],
            READONLY_DIR + 'keras/models'
        )

    if not os.path.exists(READONLY_DIR + 'week3/102flowers.tgz'):
        # originals:
        # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
        # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
        util_download.sequential_downloader(
            'https://github.com/hse-aml/intro-to-dl',
            'v0.3',
            [
                '102flowers.tgz',
                'imagelabels.mat'
            ],
            READONLY_DIR + 'week3'
        )

    util_download.link_all_files_from_dir(READONLY_DIR + 'week3/', target_path)

    # list all files in tar sorted by name
    all_files = sorted(get_all_filenames(os.path.join(target_path, '102flowers.tgz')))

    # read class labels (0, 1, 2, ...)
    all_labels = scipy.io.loadmat(os.path.join(target_path, 'imagelabels.mat'))['labels'][0] - 1

    n_classes = len(np.unique(all_labels))

    # split into train/test
    train_files, test_files, train_labels, test_labels = \
        train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    return train_files, test_files, train_labels, test_labels, n_classes


# noinspection SpellCheckingInspection,PyUnresolvedReferences
def load_lfw_dataset(use_raw=False, dx=80, dy=80, dimx=45, dimy=45):
    """
    Labeled Faces in the Wild is a database of face photographs designed for studying the
    problem of unconstrained face recognition. The data set contains more than 13,000 images
    of faces collected from the web. Each face has been labeled with the name of the person
    pictured. 1680 of the people pictured have two or more distinct photos in the data set.

    :param use_raw:
    :param dx:
    :param dy:
    :param dimx:
    :param dimy:
    :return:
    """
    # read attrs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple, df_attrs[['person', 'imagenum']].values))

    # read photos
    all_photos = []
    photo_ids = []

    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm.tqdm_notebook(f.getmembers()):
            if m.isfile() and m.name.endswith('.jpg'):
                # prepare image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))

                # parse person
                fname = os.path.split(m.name)[-1]
                fname_split = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(fname_split[:-1])
                photo_number = int(fname_split[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person': person_id, 'imagenum': photo_number})

    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # preserve photo_ids order
    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(['person', 'imagenum'], axis=1)
    return all_photos, all_attrs


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


def load_names():
    """
    The dataset contains around 8,000 names from different cultures,
    all in latin transcript.

    https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/week5/names

    :return:
    """
    with open(DATA_DIR + 'names.txt') as f:
        names = f.read()[:-1].split('\n')
        return [' ' + name for name in names]


def load_tagged_sentences():
    nltk.download('brown')
    nltk.download('universal_tagset')
    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    all_tags = ['#EOS#', '#UNK#', 'ADV', 'NOUN', 'ADP', 'PRON', 'DET', '.', 'PRT', 'VERB', 'X', 'NUM', 'CONJ', 'ADJ']
    data = np.array([[(word.lower(), tag) for word, tag in sentence] for sentence in data])
    return data, all_tags


def load_quickdraw_dataset(category, target_dir, fmt='npy'):
    """
    The Quick Draw Dataset is a collection of 50 million drawings across 345 categories,
    contributed by players of the game Quick, Draw!.

    https://github.com/googlecreativelab/quickdraw-dataset

    :param category:
    :param target_dir:
    :param fmt:
    :return: path to downloaded file
    """
    filename = '{}.{}'.format(category.replace(' ', '_'), fmt)
    file_path = os.path.join(target_dir, filename)
    folder = None
    if fmt == 'npy':
        folder = 'numpy_bitmap'

    url_template = 'https://storage.googleapis.com/quickdraw_dataset/full/{}/{}.{}'
    url = url_template.format(folder, url_quote(category), fmt)
    util_download.download_file(url, file_path)
    return file_path


def load_stack_overflow_dataset():
    """
    Posted titles from StackOverflow. All corpora (except for test) contain titles
    of the posts and corresponding tags (100 tags are available).

    :return:
    """
    if not os.path.exists(DATA_DIR + 'stackoverflow/train.tsv'):
        util_download.sequential_downloader(
            'https://github.com/hse-aml/natural-language-processing',
            'week1',
            [
                'train.tsv',
                'validation.tsv',
                'test.tsv',
                'text_prepare_tests.tsv'
            ],
            DATA_DIR + 'stackoverflow'
        )

    def read_data(filename):
        data = pd.read_csv(filename, sep='\t')
        data['tags'] = data['tags'].apply(literal_eval)
        return data

    train = read_data(DATA_DIR + 'stackoverflow/train.tsv')
    val = read_data(DATA_DIR + 'stackoverflow/validation.tsv')
    test = pd.read_csv(DATA_DIR + 'stackoverflow/test.tsv', sep='\t')

    x_train = train['title'].values
    y_train = train['tags'].values
    x_val = val['title'].values
    y_val = val['tags'].values
    x_test = test['title'].values

    return x_train, y_train, x_val, y_val, x_test
