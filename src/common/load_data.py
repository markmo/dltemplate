from ast import literal_eval
from collections import defaultdict
from common import util_download
from common.util import build_vocab, clean_text, get_all_filenames, pad_sentences
import cv2
import gensim
import h5py
import nltk
import numpy as np
import os
import pandas as pd
import scipy.io as spio
from sklearn.model_selection import train_test_split
import tarfile
from tqdm import tqdm
from urllib.parse import quote as url_quote
from zipfile import ZipFile


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


def load_cnn_daily_mail_dataset():
    """
    DeepMind Q&A Dataset.
    Hermann et al. (2015) created two awesome datasets using news articles for Q&A research.
    Each dataset contains many documents (90k and 197k each), and each document contains on
    average 4 questions approximately. Each question is a sentence with one missing word/phrase
    which can be found from the accompanying document/context.

    https://cs.nyu.edu/~kcho/DMQA/

    :return: None
    """
    target_path = DATA_DIR + 'cnn_daily_mail/'
    url = 'https://drive.google.com/file/d/0BzQ6rtO2VN95a0c3TlZCWkl3aU0/view?usp=sharing'
    filename = 'finished_files.zip'
    if not os.path.exists(target_path + 'train.bin'):
        os.makedirs(target_path, exist_ok=True)
        target_file = target_path + filename
        util_download.download_file(url, target_file)
        zip_ref = ZipFile(target_file, 'r')
        zip_ref.extractall(target_path)
        zip_ref.close()
    else:
        print('CNN Daily Mail Dataset already exists')


# noinspection SpellCheckingInspection
def load_crime_dataset():
    """
    This dataset contains incidents derived from SFPD Crime Incident Reporting system.
    The data ranges from 1/1/2003 to 5/13/2015. The training set and test set rotate
    every week, meaning week 1,3,5,7... belong to test set, week 2,4,6,8 belong to
    training set.

    Data fields
    Dates - timestamp of the crime incident
    Category - category of the crime incident (only in train.csv). This is the target
               variable you are going to predict.
    Descript - detailed description of the crime incident (only in train.csv)
    DayOfWeek - the day of the week
    PdDistrict - name of the Police Department District
    Resolution - how the crime incident was resolved (only in train.csv)
    Address - the approximate street address of the crime incident
    X - Longitude
    Y - Latitude

    :return:
    """
    filename = DATA_DIR + 'text_classification/crime/train.csv.zip'
    df = pd.read_csv(filename, compression='zip')
    selected = ['Category', 'Descript']
    x, y, vocab, vocab_inv, df, labels = prepare_classification_training_set(df, selected)
    return x, y, vocab, vocab_inv, df, labels


def prepare_classification_training_set(df, selected):
    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    labels = sorted(list(set(df[selected[0]].tolist())))
    n_labels = len(labels)
    one_hot = np.zeros((n_labels, n_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    x_raw = df[selected[1]].apply(lambda w: clean_text(w).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda w: label_dict[w]).tolist()
    x_raw = pad_sentences(x_raw)
    vocab, vocab_inv = build_vocab(x_raw)
    x = np.array([[vocab[word] for word in sent] for sent in x_raw])
    y = np.array(y_raw)
    return x, y, vocab, vocab_inv, df, labels


# noinspection SpellCheckingInspection
def load_crime_test_dataset(labels):
    filename = DATA_DIR + 'text_classification/crime/small_samples.csv'
    df = pd.read_csv(filename, sep='|')
    selected = ['Descript']
    test_examples, y, df = prepare_classification_test_set(df, selected, labels, label_colname='Category')
    return test_examples, y, df


def prepare_classification_test_set(df, selected, labels, label_colname=None):
    df = df.dropna(axis=0, how='any', subset=selected)
    test_examples = df[selected[0]].apply(lambda x: clean_text(x).split(' ')).tolist()
    n_labels = len(labels)
    one_hot = np.zeros((n_labels, n_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    y = None
    if label_colname in df.columns:
        selected.append(label_colname)
        y = df[selected[1]].apply(lambda x: label_dict[x]).tolist()

    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1)
    return test_examples, y, df


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
    all_labels = spio.loadmat(os.path.join(target_path, 'imagelabels.mat'))['labels'][0] - 1

    n_classes = len(np.unique(all_labels))

    # split into train/test
    train_files, test_files, train_labels, test_labels = \
        train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    return train_files, test_files, train_labels, test_labels, n_classes


def load_hebrew_to_english_dataset(mode='he-to-en', easy_mode=True):
    """
    Create dataset as a dict {word1: [trans1, trans2, ...], word2: [...], ...}

    Many words have several correct translations.

    :param mode:
    :param easy_mode:
    :return:
    """
    word_to_translation = defaultdict(list)
    bos = '_'
    eos = ';'
    with open(DATA_DIR + 'hebrew_translation/main_dataset.txt') as f:
        for line in f:
            en, he = line[:-1].lower().replace(bos, ' ').replace(eos, ' ').split('\t')
            word, trans = (he, en) if mode == 'he-to-en' else (en, he)
            if len(word) < 3:
                continue

            if easy_mode and max(len(word), len(trans)) > 20:
                continue

            word_to_translation[word].append(trans)

    print('Size of Hebrew-to-English translation dict:', len(word_to_translation))
    return word_to_translation


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
        for m in tqdm(f.getmembers()):
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


def load_pix2pix_dataset(category):
    """
    pix2pix datasets:

    'facades': 400 images from CMP Facades dataset <http://cmp.felk.cvut.cz/~tylecr1/facade/>.
    'cityscapes': 2975 images from the Cityscapes training set <https://www.cityscapes-dataset.com/>.
    'maps': 1096 training images scraped from Google Maps
    'edges2shoes': 50k training images from UT Zappos50K dataset
                   <http://vision.cs.utexas.edu/projects/finegrained/utzap50k/>.
    'edges2handbags': 137K Amazon Handbag images from iGAN project <https://github.com/junyanz/iGAN>.

    :param category:
    :return:
    """
    category = category.lower()
    assert category in ['facades', 'cityscapes', 'maps', 'edges2shoes', 'edges2handbags'], \
        "category arg must be one of 'facades', 'cityscapes', 'maps', 'edges2shoes', 'edges2handbags'"

    if not os.path.exists('{}pix2pix/{}'.format(DATA_DIR, category)):
        util_download.sequential_downloader(
            'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz'.format(category),
            'week1',
            [
                'train.tsv',
                'validation.tsv',
                'test.tsv',
                'text_prepare_tests.tsv'
            ],
            DATA_DIR + 'stackoverflow'
        )


def load_question_pairs_dataset(test_size=1000):
    train_df = pd.read_csv(DATA_DIR + 'question_pairs/train.csv', header=0)
    test_df = pd.read_csv(DATA_DIR + 'question_pairs/test.csv', header=0)
    return (train_df[['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']],
            test_df[['question1', 'question2']][:test_size])


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


def load_signs_dataset():
    train = h5py.File(DATA_DIR + 'signs/train_signs.h5', 'r')
    test = h5py.File(DATA_DIR + 'signs/test_signs.h5', 'r')
    x_train = np.array(train['train_set_x']).astype('float32')
    y_train = np.array(train['train_set_y'])
    x_test = np.array(test['test_set_x']).astype('float32')
    y_test = np.array(test['test_set_y'])
    classes = np.array(test['list_classes'])
    return x_train, y_train, x_test, y_test, classes


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


def load_stack_overflow_questions_dataset():
    """
    Predefined corpora. All files are tab-separated, but have different
    formats:

    * train - contains similar sentences at the same row
    * validation - contains the following columns: question, similar
      question (positive example), negative example 1, negative example 2, ...
    * test - contains the following columns: question, example 1, example 2, ...
    :return:
    """
    if not os.path.exists(DATA_DIR + 'questions/train.tsv'):
        util_download.sequential_downloader(
            'https://github.com/hse-aml/natural-language-processing',
            'week3',
            [
                'train.tsv',
                'validation.tsv',
                'test.tsv',
                'test_embeddings.tsv'
            ],
            DATA_DIR + 'questions'
        )

    def read_corpus(filename):
        data = []
        for line in open(filename, encoding='utf-8'):
            data.append(line.strip().split('\t'))

        return data

    train = read_corpus(DATA_DIR + 'questions/train.tsv')
    val = read_corpus(DATA_DIR + 'questions/validation.tsv')
    test = read_corpus(DATA_DIR + 'questions/test.tsv')

    return train, val, test


URL_TOKEN = '<URL>'
USR_TOKEN = '<USR>'


def load_tagged_sentences():
    nltk.download('brown')
    nltk.download('universal_tagset')
    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    all_tags = ['#EOS#', '#UNK#', 'ADV', 'NOUN', 'ADP', 'PRON', 'DET', '.', 'PRT', 'VERB', 'X', 'NUM', 'CONJ', 'ADJ']
    data = np.array([[(word.lower(), tag) for word, tag in sentence] for sentence in data])
    return data, all_tags


def load_twitter_entities_dataset():
    """
    Contains tweets with named-entity tags. Every line of a file contains
    a pair of a token (word/punctuation symbol) and a tag, separated by
    whitespace. Different tweets are separated by an empty line.

    :return:
    """
    # nltk.download('averaged_perceptron_tagger')
    if not os.path.exists(DATA_DIR + 'twitter/train.txt'):
        util_download.sequential_downloader(
            'https://github.com/hse-aml/natural-language-processing',
            'week2',
            [
                'train.txt',
                'validation.txt',
                'test.txt'
            ],
            DATA_DIR + 'twitter'
        )

    def read_data(filename):
        tokens = []
        tags = []
        tweet_tokens = []
        tweet_tags = []
        for line in open(filename, encoding='utf-8'):
            line = line.strip()
            if not line:
                if tweet_tokens:
                    tokens.append(tweet_tokens)
                    tags.append(tweet_tags)

                tweet_tokens = []
                tweet_tags = []
            else:
                token, tag = line.split()
                if token.startswith('http://') or token.startswith('https://'):
                    token = URL_TOKEN
                elif token.startswith('@'):
                    token = USR_TOKEN

                tweet_tokens.append(token)
                tweet_tags.append(tag)

        return tokens, tags

    tokens_train, tags_train = read_data(DATA_DIR + 'twitter/train.txt')
    tokens_val, tags_val = read_data(DATA_DIR + 'twitter/validation.txt')
    tokens_test, tags_test = read_data(DATA_DIR + 'twitter/test.txt')

    return tokens_train, tags_train, tokens_val, tags_val, tokens_test, tags_test


def load_word2vec_embeddings(limit=500000):
    """
    Pre-trained word vectors from Google, trained on a part of Google News
    dataset (about 100 billion words). The model contains 300-dimensional
    vectors for 3 million words and phrases.

    https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

    :param limit:
    :return:
    """
    url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing'
    file_path = DATA_DIR + 'word2vec/GoogleNews-vectors-negative300.bin'
    if not os.path.exists(file_path):
        util_download.download_file(url, file_path)

    return gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True, limit=limit)
