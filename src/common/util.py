from collections import Counter
import cv2
from itertools import chain, cycle
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from queue import Empty, Full, Queue
import re
from scipy import interp
from sklearn.metrics import auc, roc_curve
import tarfile
import threading
from tqdm import tqdm
import yaml
from zipfile import ZipFile

logging.getLogger().setLevel(logging.INFO)


# Common utility functions

def apply_gaussian_noise(x, sigma=0.1):
    """
    Adds noise from standard normal distribution with standard deviation sigma

    :param x:
    :param sigma:
    :return:
    """
    return x + np.random.normal(0, sigma, x.shape)


def apply_model(zip_filename, model, preprocess_for_model, extensions=('.jpg',), input_shape=(224, 224), batch_size=32):
    # queue for cropped images
    q = Queue(maxsize=(batch_size * 10))

    # when read thread, put all images in queue
    read_thread_completed = threading.Event()

    # time for read thread to die
    kill_read_thread = threading.Event()

    def reading_thread(zip_fname):
        zf = ZipFile(zip_fname)
        for fname in tqdm(zf.namelist()):
            if kill_read_thread.is_set():
                break

            if os.path.splitext(fname)[-1] in extensions:
                buf = zf.read(fname)  # read raw bytes from zip for fn
                img = decode_image_from_raw_bytes(buf)  # decode raw bytes
                img = crop_and_preprocess(img, input_shape, preprocess_for_model)
                while True:
                    try:
                        q.put((os.path.split(fname)[-1], img), timeout=1)  # put in queue
                    except Full:
                        if kill_read_thread.is_set():
                            break
                        continue
                    break

        read_thread_completed.set()  # all images read

    # start reading thread
    t = threading.Thread(target=reading_thread, args=(zip_filename,))
    t.daemon = True
    t.start()

    img_filenames = []
    img_embeddings = []
    batch_images = []

    def process_batch(batch_imgs):
        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_embeddings = model.predict(batch_imgs)
        img_embeddings.append(batch_embeddings)

    try:
        while True:
            try:
                filename, image = q.get(timeout=1)
            except Empty:
                if read_thread_completed.is_set():
                    break
                continue

            img_filenames.append(filename)
            batch_images.append(image)
            if len(batch_images) == batch_size:
                process_batch(batch_images)
                batch_images = []

            q.task_done()

        # process last batch
        if len(batch_images):
            process_batch(batch_images)

    finally:
        kill_read_thread.set()
        t.join()

    q.join()
    img_embeddings = np.vstack(img_embeddings)
    return img_embeddings, img_filenames


def batch_generator(items, batch_size):
    """
    Implement batch generator that yields items in batches of size batch_size.

    There's no need to shuffle input items, just chop them into batches.

    Remember about the last batch that can be smaller than batch_size!

    :param items: any iterable (list, generator, ...).
                  You should do `for item in items: ...`
                  In case of generator you can pass through your items only once!
    :param batch_size:
    :return: In output yield each batch as a list of items.
    """
    if not items:
        yield items

    batch = []
    for i, item in enumerate(items):
        batch.append(item)
        if (i + 1) % batch_size == 0:
            yield batch
            batch = []

    if batch:
        yield batch


def build_vocab(sentences):
    word_counts = Counter(chain(*sentences))
    vocab_inv = [word[0] for word in word_counts.most_common()]
    vocab = {word: i for i, word in enumerate(vocab_inv)}
    return vocab, vocab_inv


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9:(),!?'`]", ' ', text)
    text = re.sub(r' : ', ':', text)
    text = re.sub(r"'s", " 's", text)
    text = re.sub(r"'ve", " 've", text)
    text = re.sub(r"n't", " n't", text)
    text = re.sub(r"'re", " 're", text)
    text = re.sub(r"'d", " 'd", text)
    text = re.sub(r"'ll", " 'll", text)
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'!', ' ! ', text)
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\)', ' ) ', text)
    text = re.sub(r'\?', ' ? ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().lower()


def convert_to_one_hot(y, c):
    return np.eye(c)[y.reshape(-1)].T


def crop_and_preprocess(img, input_shape, preprocess_for_model):
    img = image_center_crop(img)  # take center crop
    img = cv2.resize(img, input_shape)  # resize for our model
    img = img.astype('float32')  # prepare for normalization
    img = preprocess_for_model(img)  # preprocess for model
    return img


def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_all_filenames(tar_filename):
    """
    read filenames directly from tar

    :param tar_filename:
    :return:
    """
    with tarfile.open(tar_filename) as f:
        return [m.name for m in f.getmembers() if m.isfile()]


def get_char_tokens(names):
    return list(set([ch for ch in ''.join(names)]))


def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.

    Returns [min(h, w), min(h, w), 3] output with same width and height.

    For cropping use numpy slicing.

    :param img:
    :return:
    """
    h, w = img.shape[:2]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2

    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]


def image_center_crop2(img):
    height, width = img.shape[:2]
    s = min(height, width)
    cropped_width = int((width - s) / 2)
    cropped_height = int((height - s) / 2)
    cropped_img = img[cropped_height:(cropped_height + s), cropped_width:(cropped_width + s), :]
    return cropped_img


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(value)
        return True
    except (TypeError, ValueError):
        return False


def load_hyperparams(file_path):
    with open(file_path) as f:
        return yaml.load(f)


def map_token_to_id(tokens):
    return {token: i for i, token in enumerate(tokens)}


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if value is not None:
            merged[key] = value

    return merged


def next_batch(x, y, batch_size):
    """
    Extract mini batches from input and matching labels

    :param x data samples
    :param y data labels
    :param batch_size size of batch
    """
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    x_shuffle = [x[i] for i in idx]
    y_shuffle = [y[i] for i in idx]
    return np.asarray(x_shuffle), np.asarray(y_shuffle)


def one_hot_encode(labels, n_classes):
    n_samples = labels.shape[0]
    encoded = np.zeros((n_samples, n_classes))
    encoded[np.arange(n_samples), labels] = 1
    return encoded


def pad_sentences(sentences, pad_token='<PAD>', forced_seq_len=None):
    if forced_seq_len is None:  # Train
        seq_len = max(len(sent) for sent in sentences)
    else:  # Prediction
        logging.info('In prediction, reading trained sequence length...')
        seq_len = forced_seq_len

    logging.info('Max sequence length: {}'.format(seq_len))
    padded_sentences = []
    for sent in sentences:
        n_pad = seq_len - len(sent)
        if n_pad < 0:
            # In prediction, cut off sentence if it is longer than sequence length
            logging.info('Sentence truncated because it is longer than trained sequence length')
            padded_sent = sent[0: seq_len]
        else:
            padded_sent = sent + [pad_token] * n_pad

        padded_sentences.append(padded_sent)

    return padded_sentences


def plot_accuracy(n_epochs, train_costs, val_costs):
    iterations = list(range(n_epochs))
    plt.figure()
    plt.plot(iterations, train_costs, label='Train')
    plt.plot(iterations, val_costs, label='Valid')
    plt.legend()
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()


# noinspection SpellCheckingInspection
def plot_roc_auc(y_test, y_score, n_classes):
    """
    Plots ROC curve for macro and micro averaging.

    :param y_test:
    :param y_score:
    :param n_classes:
    :return:
    """
    # compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # plot all ROC curves
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc['macro']),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(0, 3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of ROC to multi-class')
    plt.legend(loc='lower right')
    plt.show()


def random_minibatches(x, y, batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (x, y)

    :param x: input data of shape (number of examples, image height, image width, number of channels)
    :param y: true "label" vector of shape (number of examples, number of classes)
    :param batch_size: integer, minibatch size
    :param seed: random seed
    :return: list of synchronous (minibatch_x, minibatch_y)
    """
    # print('x shape:', x.shape)
    # print('y shape:', y.shape)

    np.random.seed(seed)
    m = x.shape[0]  # number of training examples
    minibatches = []

    # Step 1: Shuffle (x, y)
    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_y), minus the end case
    n_complete_minibatches = int(math.floor(m / batch_size))
    for k in range(0, n_complete_minibatches):
        minibatch_x = shuffled_x[k * batch_size:(k + 1) * batch_size, :, :, :]
        minibatch_y = shuffled_y[k * batch_size:(k + 1) * batch_size, :]
        minibatch = (minibatch_x, minibatch_y)
        minibatches.append(minibatch)

    # Handling the end case (last minibatch < batch_size)
    if m % batch_size != 0:
        minibatch_x = shuffled_x[n_complete_minibatches * batch_size:, :, :, :]
        minibatch_y = shuffled_y[n_complete_minibatches * batch_size:, :]
        minibatch = (minibatch_x, minibatch_y)
        minibatches.append(minibatch)

    # (minibatch_x, minibatch_y) = minibatches[0]
    # print('minibatch_x shape:', minibatch_x.shape)
    # print('minibatch_y shape:', minibatch_y.shape)

    return minibatches


def raw_generator_with_label_from_tar(tar_filename, files, labels):
    label_by_filename = dict(zip(files, labels))
    with tarfile.open(tar_filename) as f:
        while True:
            m = f.next()
            if m is None:
                break

            if m.name in label_by_filename:
                yield f.extractfile(m).read(), label_by_filename[m.name]


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def read_raw_from_tar(tar_filename, filename):
    """
    reads bytes directly from tar by filename
    (slow, but ok for testing, takes ~6 sec)

    :param tar_filename:
    :param filename:
    :return:
    """
    with tarfile.open(tar_filename) as f:
        m = f.getmember(filename)
        return f.extractfile(m).read()


def reshape(x, n, dtype='float32'):
    return x.reshape(x.shape[0], n).astype(dtype)


def sample_zip(filename_in, filename_out, rate=0.01, seed=42):
    np.random.seed(seed)
    with ZipFile(filename_in) as fin, ZipFile(filename_out, 'w') as f:
        sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
        for zip_info in sampled:
            f.writestr(zip_info, fin.read(zip_info))


def save_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def to_token_id_matrix(names, token_to_id, max_len=None, pad=0, dtype='int32'):
    """
    Converts a list of names into an RNN-digestible matrix

    :param names:
    :param token_to_id:
    :param max_len:
    :param pad:
    :param dtype:
    :return:
    """
    max_len = max_len or max(map(len, names))
    n = len(names)
    matrix = np.zeros([n, max_len], dtype) + pad
    for i in range(n):
        idx = list(map(token_to_id.get, names[i]))
        matrix[i, :len(idx)] = idx

    return matrix.T


def zipdir(path, ziph):
    # ziph is zipfile handle
    length = len(path)
    for root, dirs, files in os.walk(path):
        folder = root[length:]  # path without "parent"
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(folder, file))
