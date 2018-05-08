import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from queue import Empty, Full, Queue
import tarfile
import threading
import tqdm
from zipfile import ZipFile


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
        for fname in tqdm.tqdm_notebook(zf.namelist()):
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


def plot_accuracy(n_epochs, train_costs, val_costs):
    iterations = list(range(n_epochs))
    plt.figure()
    plt.plot(iterations, train_costs, label='Train')
    plt.plot(iterations, val_costs, label='Valid')
    plt.legend()
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()


def prepare_raw_bytes_for_model(raw_bytes, img_size, normalize_for_model=True):
    img = decode_image_from_raw_bytes(raw_bytes)  # decode image raw bytes to matrix
    img = image_center_crop(img)  # take squared center crop
    img = cv2.resize(img, (img_size, img_size))  # resize for our model
    if normalize_for_model:
        img = img.astype('float32')  # prepare for normalization
        img = keras.applications.inception_v3.preprocess_input(img)  # normalize for model

    return img


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
    with ZipFile(filename_in) as fin, ZipFile(filename_out, 'w') as fout:
        sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
        for zip_info in sampled:
            fout.writestr(zip_info, fin.read(zip_info))


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
