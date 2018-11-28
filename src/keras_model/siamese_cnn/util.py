from contextlib import contextmanager
import gensim
import keras
import math
import numpy as np
import os
import pandas as pd
import pickle
import spacy
import sys
import time

PAD = '__PAD__'
UNK = '__UNK__'


@contextmanager
def timed(activity):
    print('\nStart', activity)
    tic = time.time()
    yield
    toc = time.time()
    print('End {}, duration: {:.5f}'.format(activity, toc - tic))


def prepare_training_data(train_df):
    if os.path.exists('doc1_tokens.npy'):
        with timed('loading tokens'):
            q1_tokens = np.load('doc1_tokens.npy')
            q2_tokens = np.load('doc2_tokens.npy')
            q1_length = np.vectorize(len)(q1_tokens)
            q2_length = np.vectorize(len)(q1_tokens)
    else:
        with timed('loading spacy'):
            tokenize = spacy.load('en_core_web_sm')

        with timed('tokenizing'):
            q1_tokens = train_df.question1.apply(lambda q: [t.text for t in tokenize(q)]).values
            q2_tokens = train_df.question2.apply(lambda q: [t.text for t in tokenize(q)]).values
            q1_length = np.vectorize(len)(q1_tokens)
            q2_length = np.vectorize(len)(q1_tokens)

    return train_df.assign(q1_tokens=q1_tokens, q2_tokens=q2_tokens, q1_length=q1_length, q2_length=q2_length)


def create_vocab(train_df):
    with timed('creating vocab'):
        docs = np.concatenate([train_df.q1_tokens, train_df.q2_tokens])
        max_length = 0
        unique_tokens = set()
        for doc in docs:
            n_tokens = len(doc)
            if n_tokens > max_length:
                max_length = n_tokens

            for token in doc:
                unique_tokens.add(token)

        word2idx = {word: i + 2 for i, word in enumerate(unique_tokens)}
        word2idx[PAD] = 0
        word2idx[UNK] = 1
        idx2word = {i: word for word, i in word2idx.items()}
        print('Vocab size: {:,}'.format(len(word2idx)))

        return word2idx, idx2word


def encode(train_df, word2idx):
    with timed('encoding'):
        q1_encoded = train_df.q1_tokens.apply(lambda xs: [word2idx[x] for x in xs]).values
        q2_encoded = train_df.q2_tokens.apply(lambda xs: [word2idx[x] for x in xs]).values

    return train_df.assign(q1_encoded=q1_encoded, q2_encoded=q2_encoded)

    # Shuffle data
    # np.random.seed(42)
    # shuffle_indices = np.random.permutation(np.arange(len(train_df)))
    # docs1_shuffled = docs1[shuffle_indices]
    # docs2_shuffled = docs2[shuffle_indices]
    # y_shuffled = train_df.is_duplicate[shuffle_indices].values
    #
    # return docs1_shuffled, docs2_shuffled, y_shuffled, train_df


def load_embeddings(idx2word, embed_size, word2vec_filename):
    with timed('loading word2vec'):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_filename, binary=True, limit=500000)
        print('word2vec shape:', word2vec.shape)

    with timed('building embeddings'):
        embedding_lookup = {}
        for word, vector in zip(word2vec.vocab, word2vec.vectors):
            embedding_lookup[word] = vector

        vocab_size = len(idx2word)
        bound = np.sqrt(6.0) / np.sqrt(vocab_size)
        embeddings = np.zeros([vocab_size, embed_size])
        print('embeddings shape:', embeddings.shape)
        embeddings[0] = np.zeros(embed_size)
        for i in range(1, vocab_size):
            embedding = embedding_lookup.get(idx2word[i], None)
            if embedding is None:
                embeddings[i] = np.random.uniform(-bound, bound, embed_size)
            else:
                embeddings[i] = embedding

        return embeddings


class BucketedBatchGenerator(keras.utils.Sequence):

    def __init__(self, input_data, batch_size, split, max_doc1_length, max_doc2_length, shuffle=True, gpus=0):
        super().__init__()
        self.input_data = input_data
        self.batch_size = batch_size
        self.split = split
        self.max_doc1_length = max_doc1_length
        self.max_doc2_length = max_doc2_length
        self.shuffle = shuffle
        self.gpus = gpus
        self.buckets = list(self.input_data[self.split].keys())
        self.bucket_lengths = {b: len(self.input_data[self.split][b]['docs2']) for b in self.buckets}
        self.bucket_batches = self.get_bucket_batches()
        self.length = sum(self.bucket_batches.values())
        self.bucket_indexes = {b: list(range(self.bucket_lengths[b])) for b in self.buckets}
        self.bucket_starts = self.get_bucket_starts()
        self.reset_indexes()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        bucket, i = self.get_bucket_and_index(item)
        if i * self.batch_size + self.batch_size >= len(self.bucket_indexes[bucket]):
            batch_indexes = self.bucket_indexes[bucket][i * self.batch_size:]
        else:
            batch_indexes = self.bucket_indexes[bucket][i * self.batch_size:i * self.batch_size + self.batch_size]

        docs1 = list_encoded_arrays_to_matrix_with_padding(
            [self.input_data['docs1'][x]
             for x in self.input_data[self.split][bucket]['docs1'][batch_indexes]], self.max_doc1_length)
        docs2 = list_encoded_arrays_to_matrix_with_padding(
            [self.input_data['docs2'][x]
             for x in self.input_data[self.split][bucket]['docs2'][batch_indexes]], self.max_doc2_length)
        outcomes = [self.input_data[self.split][bucket]['outcomes'][x] for x in batch_indexes]
        return [docs1, docs2], np.array(outcomes)

    def on_epoch_end(self):
        self.reset_indexes()

    def get_bucket_batches(self):
        bucket_batches = {}
        for b in self.buckets:
            remainder = self.bucket_lengths[b] % self.batch_size
            if 0 < remainder < self.gpus:
                bucket_batches[b] = math.floor(self.bucket_lengths[b] / self.batch_size)
            else:
                bucket_batches[b] = math.ceil(self.bucket_lengths[b] / self.batch_size)

        return bucket_batches

    def get_bucket_starts(self):
        start = 0
        bucket_starts = {}
        for b in self.buckets:
            bucket_starts[b] = start
            start += self.bucket_batches[b]

        return bucket_starts

    def reset_indexes(self):
        if self.shuffle:
            np.random.shuffle(self.buckets)
            self.bucket_starts = self.get_bucket_starts()
            for b in self.buckets:
                np.random.shuffle(self.bucket_indexes[b])

    def get_bucket_and_index(self, idx):
        for b in reversed(self.buckets):
            if idx >= self.bucket_starts[b]:
                return b, idx - self.bucket_starts[b]

        return None  # should never get here


def list_encoded_arrays_to_matrix_with_padding(encoded_list, max_length):
    truncated_list = [np.resize(x, max_length) if x.shape[0] > max_length else x for x in encoded_list]
    pad_to = max(truncated_list, key=lambda x: x.size).size
    return np.vstack([np.pad(x, (0, pad_to - len(x)), mode='constant').reshape(pad_to, 1).T for x in truncated_list])


def bucket_cases(train_df, n_doc1_quantile, n_doc2_quantile):
    doc1_quantile_labels = ['q1_' + str(x) for x in range(1, n_doc1_quantile + 1)]
    doc1_quantiles = [float(x) / n_doc1_quantile for x in range(0, n_doc1_quantile + 1)]
    doc2_quantile_labels = ['q2_' + str(x) for x in range(1, n_doc2_quantile + 1)]
    doc2_quantiles = [float(x) / n_doc2_quantile for x in range(0, n_doc2_quantile + 1)]
    train_df = train_df.assign(q1_quantiles=pd.qcut(train_df.q1_length, q=doc1_quantiles, labels=doc1_quantile_labels))
    for ql in doc1_quantile_labels:
        train_df.loc[train_df.q1_quantiles == ql, 'q2_quantiles'] = \
            pd.qcut(train_df[train_df.q1_quantiles == ql].q2_length, q=doc2_quantiles, labels=doc2_quantile_labels)

    return train_df.assign(bucket=train_df.apply(lambda x: '{}_{}'.format(x.q2_quantiles, x.q1_quantiles), axis=1))


def write_bucket(df, dirname, bucket_id=None):
    bucket_dir = dirname
    if not bucket_dir.endswith('/'):
        bucket_dir += '/'

    if bucket_id:
        bucket_dir += bucket_id.replace(' ', '_') + '/'
    else:
        bucket_dir += 'all/'

    try:
        os.makedirs(bucket_dir, exist_ok=True)
    except OSError as e:
        print('Error writing encoded arrays:', e, file=sys.stderr)
        return

    np.save(bucket_dir + 'docs1.npy', encoded_series_to_matrix_with_padding(df.q1_encoded, df.q1_length.max()))
    np.save(bucket_dir + 'docs2.npy', encoded_series_to_matrix_with_padding(df.q2_encoded, df.q2_length.max()))
    np.save(bucket_dir + 'outcomes.npy', ((df.is_duplicate is True) * 1).values.reshape(-1, 1))
    np.save(bucket_dir + 'doc1_ids.npy', df.q1id.values.reshape(-1, 1))
    np.save(bucket_dir + 'doc2_ids.npy', df.q2id.values.reshape(-1, 1))
    print('\rSaving numpy arrays to {dir:{width}}'.format(dir=bucket_dir, width=len(bucket_dir) + 10), end='')


def encoded_series_to_matrix_with_padding(encoded_series, pad_to):
    print('\nencoded_series shape:', encoded_series.shape)
    print('pad_to:', pad_to)
    return np.vstack([np.pad(x, (0, pad_to - len(x)), mode='constant').reshape(pad_to, 1).T for x in encoded_series])


def load_bucketed_data(input_dir):
    data = {}
    for split in ['train', 'dev']:
        data[split] = {}
        for b in os.listdir(input_dir + '/' + split + '/'):
            data[split][b] = {}
            bucket_dir = input_dir + '/' + split + '/' + b + '/'
            try:
                data[split][b]['docs1'] = np.load(bucket_dir + 'docs1.npy')
                data[split][b]['docs2'] = np.load(bucket_dir + 'docs2.npy')
                data[split][b]['outcomes'] = np.load(bucket_dir + 'outcomes.npy')
            except IOError as e:
                print("Err: can't load data;", e)

    data['docs1'] = pickle.load(open(input_dir + '/docs1.pkl', 'rb'))
    data['docs2'] = pickle.load(open(input_dir + '/docs2.pkl', 'rb'))
    return data


def fully_padded_batch(input_data, split, max_doc1_length, max_doc2_length):
    docs1 = []
    docs2 = []
    outcomes = []
    for b in input_data[split].keys():
        docs1.extend([input_data['docs1'][x] for x in input_data[split][b]['docs1']])
        docs2.extend([input_data['docs2'][x] for x in input_data[split][b]['docs2']])
        outcomes.extend(input_data[split][b]['outcomes'].tolist())

    docs1_array = list_encoded_arrays_to_matrix_with_padding(docs1, max_doc1_length)
    docs2_array = list_encoded_arrays_to_matrix_with_padding(docs2, max_doc2_length)
    return [docs1_array, docs2_array], outcomes
