from argparse import ArgumentParser
from common.model_util import load_hyperparams, merge_dict
from keras.callbacks import ModelCheckpoint
from keras_model.decomposable_attention.model_setup import decomposable_attention
from keras_model.decomposable_attention.util import BatchGenerator
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/data/')
PAD = '__PAD__'
UNK = '__UNK__'


def load_question_pairs_dataset(test_size=1000):
    train_df = pd.read_csv(DATA_DIR + 'question_pairs/train_med.csv', header=0)
    test_df = pd.read_csv(DATA_DIR + 'question_pairs/test.csv', header=0)
    return (train_df[['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']],
            test_df[['question1', 'question2']][:test_size])


def create_data_file(train_df):
    q1 = train_df.question1.values
    q2 = train_df.question2.values
    combined = np.concatenate((q1, q2))
    filename = os.path.join(os.path.dirname(__file__), 'data.txt')
    np.savetxt(filename, combined, fmt='%s')


def load_embeddings_dict(embed_filename):
    embeddings = []
    embeddings_dict = {}
    word2idx = {PAD: 0, UNK: 1}
    with open(embed_filename, 'r') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            word2idx[word] = i + 2
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs
            embeddings.append(coefs)

    return np.array(embeddings), embeddings_dict, word2idx


def convert_to_onehot(x_raw, word2idx):
    unk_idx = word2idx[UNK]
    encoded = []
    for line in x_raw:
        words = re.split(r'\s+', line.strip())
        onehot = [word2idx.get(w, unk_idx) for w in words]
        encoded.append(onehot)

    return encoded


def pad(x_encoded, max_len, pad_idx):
    n = len(x_encoded)
    if n < max_len:
        padded = x_encoded + [pad_idx] * (max_len - n)
    else:
        padded = x_encoded[:max_len]

    return padded


def preprocess(train_df, word2idx, max_len):
    q1 = train_df.question1.values
    q2 = train_df.question2.values

    # Don't need to pad as using an RNN
    # pad_idx = word2idx[PAD]
    # q1_encoded = pad(convert_to_onehot(q1, word2idx), max_len, pad_idx)
    # q2_encoded = pad(convert_to_onehot(q2, word2idx), max_len, pad_idx)

    q1_encoded = convert_to_onehot(q1, word2idx)
    q2_encoded = convert_to_onehot(q2, word2idx)
    return train_df.assign(q1_encoded=q1_encoded, q2_encoded=q2_encoded)


def run(constant_overwrites):
    print('Running')
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    max_len = 30
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint')
    n_epochs = constants['n_epochs']
    batch_size = constants['batch_size']

    embeddings, _, word2idx = load_embeddings_dict(os.path.join(os.path.dirname(__file__), 'ft.vec'))
    ft_matrix_filename = os.path.join(os.path.dirname(__file__), 'fasttext_matrix.npy')
    if not os.path.exists(ft_matrix_filename):
        np.save(ft_matrix_filename, embeddings)

    train_df, test_df = load_question_pairs_dataset()
    train_df = preprocess(train_df, word2idx, max_len)
    train_df, val_df = train_test_split(train_df, test_size=0.1, shuffle=True)
    test_df = preprocess(test_df, word2idx, max_len)

    model = decomposable_attention(ft_matrix_filename)
    print('\nModel summary:')
    print(model.summary())

    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    training_generator = BatchGenerator(train_df.q1_encoded.values, train_df.q2_encoded.values,
                                        train_df.y.values, batch_size)
    validation_data = BatchGenerator(val_df.q1_encoded.values, val_df.q2_encoded.values,
                                     val_df.y.values, batch_size)
    train_history = model.fit_generator(training_generator, steps_per_epoch=len(training_generator),
                                        epochs=n_epochs, use_multiprocessing=True, workers=6,
                                        validation_data=(validation_data), callbacks=[checkpoint],
                                        verbose=1,
                                        initial_epoch=0  # use when restarting training (*zero* based)
                                        )


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Decomposable Attention Model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--train', dest='train', help='run training', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
