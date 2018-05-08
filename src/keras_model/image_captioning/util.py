from collections import Counter, defaultdict
from common.util import apply_model, crop_and_preprocess, decode_image_from_raw_bytes, image_center_crop
from common.util import read_pickle, save_pickle
import json
from keras_model.image_captioning.model_setup import cnn_encoder_builder
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.contrib import keras
import tqdm
from zipfile import ZipFile

K = keras.backend

# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

DATA_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/data/')


def apply_model_to_image_raw_bytes(sess, raw, model, data, constants):
    img_size = constants['img_size']
    img = decode_image_from_raw_bytes(raw)
    plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = crop_and_preprocess(img, (img_size, img_size), model.preprocess_for_model)
    print(' '.join(generate_caption(sess, img, model, data, constants)[1:-1]))
    plt.show()


def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    Captions have different length, but we need to batch them. That's why
    we add PAD tokens so that all sentences have equal length.

    We crunch LSTM through all the tokens, but will ignore padding tokens
    during loss calculation.

    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]

    Put vocabulary indexed captions into an np.array of shape (len(batch_captions), columns),
    where "columns" is max(map(len, batch_captions)) when `max_len` is None
    and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.

    Add padding with `pad_idx` where necessary.

    Input example: [[1, 2, 3], [4, 5]]

    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None

    Output example: np.array([[1, 2], [4, 5]]) if max_len=2

    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100

    Using numpy for speed.

    :param batch_captions:
    :param pad_idx:
    :param max_len:
    :return:
    """
    columns = max(map(len, batch_captions))
    if max_len is not None:
        columns = min(columns, max_len)

    matrix = np.full((len(batch_captions), columns), pad_idx, dtype='float32')
    for i, tokens in enumerate(batch_captions):
        n = min(len(tokens), columns)
        matrix[i][0:n] = tokens[0:n]

    return matrix


def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]

    Use `split_sentence` function to split sentence into tokens.

    Replace all tokens with vocabulary indices. Use UNK for unknown words (out of vocabulary).

    Add START and END tokens to the start and end of each sentence.

    For the example above, you would produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"] vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"] vocab[END]],
            ...
        ],
        ...
    ]

    :param captions:
    :param vocab:
    :return:
    """
    unk_idx = vocab[UNK]
    vocab = defaultdict(lambda: unk_idx, vocab)
    res = []
    for sentences in captions:
        tokens_list = []
        for sentence in sentences:
            tokens = [vocab[START]]
            tokens.extend([vocab[token] for token in split_sentence(sentence)])
            tokens.append(vocab[END])
            tokens_list.append(tokens)

        res.append(tokens_list)

    return res


def check_after_training(n_examples, decoder, data, constants):
    batch_size = constants['batch_size']
    vocab_inverse = data['vocab_inverse']
    pad_idx = data['pad_idx']
    fd = generate_batch(data['img_embeds_train'], data['captions_indexed_train'], batch_size, decoder, pad_idx)
    logits = decoder.flat_token_logits.eval(fd)
    truth = decoder.flat_ground_truth.eval(fd)
    mask = decoder.flat_loss_mask.eval(fd).astype(bool)
    print('Loss:', decoder.loss.eval(fd))
    print('Accuracy:', accuracy_score(logits.argmax(axis=1)[mask], truth[mask]))
    for example_idx in range(n_examples):
        print('Example', example_idx)
        print('Predicted:', decode_sentence(logits.argmax(axis=1).reshape((batch_size, -1))[example_idx],
                                            vocab_inverse))
        print('Truth:', decode_sentence(truth.reshape((batch_size, -1))[example_idx], vocab_inverse))
        print('')


def decode_sentence(sentence_indices, vocab_inverse):
    return ' '.join(list(map(vocab_inverse.get, sentence_indices)))


def generate_batch(image_embeddings, indexed_captions, batch_size, decoder, pad_idx, max_len=None):
    """
    Generate batch via random sampling of images and captions. Use `max_len`
    parameter to control the length of caption (truncating long captions).

    `image_embeddings` is an np.array of shape [number of images, img_embed_size].

    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.

    Take random images and choose one random caption for each image.

    Remember to use `batch_captions_to_matrix` for padding and respect `max_len` parameter.

    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.

    :param image_embeddings:
    :param indexed_captions:
    :param batch_size:
    :param decoder
    :param pad_idx
    :param max_len:
    :return:
    """
    idxs = np.random.permutation(len(image_embeddings))
    batch_image_embeddings = image_embeddings[idxs[0:batch_size]]
    captions = [x[np.random.randint(0, 5)] for x in indexed_captions[idxs[0:batch_size]]]
    batch_captions_matrix = batch_captions_to_matrix(captions, pad_idx, max_len)

    # print('')
    # print('batch_image_embeddings.shape', np.shape(batch_image_embeddings))
    # print('batch_image_embeddings[0].shape', np.shape(batch_image_embeddings[0]))
    # print('batch_image_embeddings[1].shape', np.shape(batch_image_embeddings[1]))
    # print('batch_captions_matrix.shape', np.shape(batch_captions_matrix))
    # print('batch_captions_matrix[0].shape', np.shape(batch_captions_matrix[0]))
    # print('batch_captions_matrix[1].shape', np.shape(batch_captions_matrix[1]))

    return {decoder.img_embeds: batch_image_embeddings,
            decoder.sentences: batch_captions_matrix}


def generate_caption(sess, image, model, data, constants, t=1, sample=False):
    """
    Generate caption for a given image.

    If `sample` is True, sample next token from predicted probability distribution.

    `t` is a temperature during that sampling. Higher `t` causes more
    uniform-like distribution = more chaos.

    :param sess
    :param image:
    :param model:
    :param data:
    :param constants:
    :param t:
    :param sample:
    :return:
    """
    vocab = data['vocab']
    vocab_inverse = data['vocab_inverse']

    # condition LSTM on the image
    sess.run(model.init_lstm, {model.input_images: [image]})

    # current caption
    # start with only START token
    caption = [vocab[START]]

    for _ in range(constants['max_len']):
        next_word_probs = sess.run(model.one_step, {model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()

        # apply temperature
        # for high temperature we have more uniform distribution
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        # look at how temperature works for probability distributions
        # _ = np.array([0.5, 0.4, 0.1])
        # for t in [0.01, 0.1, 1, 10, 100]:
        #     print(' '.join(map(str, _**(1/t) / np.sum(_**(1/t)))), 'with temperature', t)

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break

    return list(map(vocab_inverse.get, caption))


def generate_vocabulary(captions_train):
    """
    Return {token: index} for all training tokens (words) that occur 5 or more times.

    `index` is from 0 to N, where N is number of unique tokens in the resulting dict.

    Use `split_sentence` function to split sentence into tokens.

    Also add the following special tokens to the vocabulary:  PAD for batch padding,
    UNK for unknown word, out of vocabulary, START for start of sentence, and END
    for end of sentence.

    :param captions_train:
    :return:
    """
    vocab = Counter()
    for sentences in captions_train:
        for sentence in sentences:
            vocab.update(split_sentence(sentence))

    vocab = [token for token, k in vocab.items() if k > 4]
    vocab.extend([PAD, UNK, START, END])
    return {token: i for i, token in enumerate(sorted(vocab))}


def get_captions(img_filenames_train, img_filenames_val):
    captions_train = load_captions(img_filenames_train,
                                   DATA_DIR + 'image_captioning/captions_train-val2014.zip',
                                   'annotations/captions_train2014.json')
    captions_val = load_captions(img_filenames_val,
                                 DATA_DIR + 'image_captioning/captions_train-val2014.zip',
                                 'annotations/captions_val2014.json')
    return {
        'captions_train': captions_train,
        'captions_val': captions_val
    }


def get_pad_idx(vocab):
    return vocab[PAD]


def load_captions(filenames, zip_filename, zip_json_path):
    zf = ZipFile(zip_filename)
    js = json.loads(zf.read(zip_json_path).decode('utf8'))
    id_to_filename = {img['id']: img['file_name'] for img in js['images']}
    filename_to_caps = defaultdict(list)
    for cap in js['annotations']:
        filename_to_caps[id_to_filename[cap['image_id']]].append(cap['caption'])

    filename_to_caps = dict(filename_to_caps)
    return list(map(lambda x: filename_to_caps[x], filenames))


def load_embeddings():
    return {
        'img_embeds_train': read_pickle(DATA_DIR + 'image_captioning/train_img_embeds.pickle'),
        'img_filenames_train': read_pickle(DATA_DIR + 'image_captioning/train_img_fns.pickle'),
        'img_embeds_val': read_pickle(DATA_DIR + 'image_captioning/val_img_embeds.pickle'),
        'img_filenames_val': read_pickle(DATA_DIR + 'image_captioning/val_img_fns.pickle')
    }


def show_training_example(image_filenames_train, captions_train, example_idx=0):
    """
    You can change example_idx and see a different image

    :param image_filenames_train:
    :param captions_train:
    :param example_idx:
    :return:
    """
    zf = ZipFile(DATA_DIR + 'image_captioning/train2014_sample.zip')
    captions_by_file = dict(zip(image_filenames_train, captions_train))
    all_files = set(image_filenames_train)
    found_files = list(filter(lambda x: x.filename.rsplit('/')[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    img = decode_image_from_raw_bytes(zf.read(example))
    plt.imshow(image_center_crop(img))
    plt.title('\n'.join(captions_by_file[example.filename.rsplit('/')[-1]]))
    plt.show()


def show_valid_example(sess, model, data, constants, example_idx=0):
    img_filenames_val = data['img_filenames_val']
    zf = ZipFile(DATA_DIR + 'image_captioning/val2014_sample.zip')
    all_files = set(img_filenames_val)
    found_files = list(filter(lambda x: x.filename.rsplit('/')[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(sess, zf.read(example), model, data, constants)


def split_sentence(sentence):
    """
    Split sentence into tokens (lowercase words)

    :param sentence:
    :return:
    """
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))


def train_cnn_encoder(constants):
    """
    Features extraction takes too much time on CPU:

    * Takes 16 minutes on GPU.
    * 25x slower (InceptionV3) on CPU and takes 7 hours.
    * 10x slower (MobileNet) on CPU and takes 3 hours.

    :param constants:
    :return:
    """
    img_size = constants['img_size']
    # load pretrained model
    K.clear_session()
    encoder, preprocess_for_model = cnn_encoder_builder()

    # extract trained features
    img_embeds_train, img_filenames_train = apply_model(DATA_DIR + 'image_captioning/train2014.zip',
                                                        encoder,
                                                        preprocess_for_model,
                                                        input_shape=(img_size, img_size))
    save_pickle(img_embeds_train, DATA_DIR + 'image_captioning/train_img_embeds.pickle')
    save_pickle(img_filenames_train, DATA_DIR + 'image_captioning/train_img_fns.pickle')

    # extract validation features
    img_embeds_val, img_filenames_val = apply_model(DATA_DIR + 'image_captioning/val2014.zip',
                                                    encoder,
                                                    preprocess_for_model,
                                                    input_shape=(img_size, img_size))
    save_pickle(img_embeds_val, DATA_DIR + 'image_captioning/val_img_embed.pickle')
    save_pickle(img_filenames_val, DATA_DIR + 'image_captioning/val_img_fns.pickle')


def train(sess, train_step, decoder, data, constants, saver, reproducible=False):
    if reproducible:
        np.random.seed(42)
        random.seed(42)

    # initialize all variables
    sess.run(tf.global_variables_initializer())

    batch_size = constants['batch_size']
    pad_idx = data['pad_idx']
    max_len = constants['max_len']
    n_batches_per_epoch = constants['n_batches_per_epoch']
    n_validation_batches = constants['n_validation_batches']

    # print('')
    # print('batch_size:', batch_size)
    # print('pad_idx:', pad_idx)
    # print('max_len:', max_len)
    # print('n_batches_per_epoch:', n_batches_per_epoch)
    # print('n_validation_batches:', n_validation_batches)

    for epoch in range(constants['n_epochs']):
        train_loss = 0
        progress_bar = tqdm.tqdm_notebook(range(n_batches_per_epoch))
        counter = 0
        for _ in progress_bar:
            img_embeds_train = data['img_embeds_train']
            captions_indexed_train = data['captions_indexed_train']

            # print('')
            # print('img_embeds_train.shape:', np.shape(img_embeds_train))
            # print('captions_indexed_train.shape:', np.shape(captions_indexed_train))

            train_loss += sess.run([decoder.loss, train_step],
                                   generate_batch(img_embeds_train,
                                                  captions_indexed_train,
                                                  batch_size,
                                                  decoder,
                                                  pad_idx,
                                                  max_len))[0]
            counter += 1
            progress_bar.set_description('Training loss: %f' % (train_loss / counter))

        train_loss /= n_batches_per_epoch

        val_loss = 0
        for _ in range(n_validation_batches):
            val_loss += sess.run(decoder.loss,
                                 generate_batch(data['img_embeds_val'],
                                                data['captions_indexed_val'],
                                                batch_size,
                                                decoder,
                                                pad_idx,
                                                max_len))

        val_loss /= n_validation_batches

        print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

        # save weights after finishing epoch
        saver.save(sess, os.path.abspath('weights_{}'.format(epoch)))

    print('Finished!')
