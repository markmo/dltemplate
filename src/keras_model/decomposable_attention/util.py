import keras
import math
import numpy as np
import os
import pandas as pd
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data/')
PAD = '__PAD__'
UNK = '__UNK__'


class BatchGenerator(keras.utils.Sequence):

    def __init__(self, x_set1, x_set2, y_set, batch_size):
        self.x1, self.x2, self.y = x_set1, x_set2, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))

    def __getitem__(self, idx):
        x1_batch = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        x2_batch = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [x1_batch, x2_batch], y_batch


class BucketedBatchGenerator(keras.utils.Sequence):

    def __init__(self, input_data, batch_size, max_doc1_length, max_doc2_length, shuffle=True, gpus=0):
        super().__init__()
        self.input_data = input_data
        self.batch_size = batch_size
        self.max_doc1_length = max_doc1_length
        self.max_doc2_length = max_doc2_length
        self.shuffle = shuffle
        self.gpus = gpus
        self.buckets = [x for x in self.input_data.keys() if x not in ['docs1', 'docs2']]
        self.bucket_lengths = {b: len(self.input_data[b]['docs2']) for b in self.buckets}
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

        docs1 = np.array([self.input_data['docs1'][x] for x in self.input_data[bucket]['docs1'][batch_indexes]])
        docs2 = np.array([self.input_data['docs2'][x] for x in self.input_data[bucket]['docs2'][batch_indexes]])
        outcomes = [self.input_data[bucket]['outcomes'][x] for x in batch_indexes]
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


# noinspection PyTypeChecker
def bucket_cases(train_df, n_doc1_quantile, n_doc2_quantile):
    doc1_quantile_labels = ['q1_' + str(x) for x in range(1, n_doc1_quantile + 1)]
    doc1_quantiles = [float(x) / n_doc1_quantile for x in range(0, n_doc1_quantile + 1)]
    doc2_quantile_labels = ['q2_' + str(x) for x in range(1, n_doc2_quantile + 1)]
    doc2_quantiles = [float(x) / n_doc2_quantile for x in range(0, n_doc2_quantile + 1)]
    train_df = train_df.assign(q1_quantiles=pd.qcut(train_df.q1_length, q=doc1_quantiles, labels=doc1_quantile_labels))
    for ql in doc1_quantile_labels:
        train_df.loc[train_df.q1_quantiles == ql, 'q1_bucket_max'] = train_df.loc[train_df.q1_quantiles == ql].max()
        train_df.loc[train_df.q1_quantiles == ql, 'q2_quantiles'] = \
            pd.qcut(train_df[train_df.q1_quantiles == ql].q2_length, q=doc2_quantiles, labels=doc2_quantile_labels)

    for ql in doc1_quantile_labels:
        # print('quartile:', ql)
        train_df.loc[train_df.q1_quantiles == ql, 'q1_bucket_max'] = \
            min(60, train_df.loc[train_df.q1_quantiles == ql].q1_length.max())

    for ql in doc2_quantile_labels:
        # print('quartile:', ql)
        train_df.loc[train_df.q2_quantiles == ql, 'q2_bucket_max'] = \
            min(60, train_df.loc[train_df.q2_quantiles == ql].q2_length.max())

    return train_df.assign(bucket=train_df.apply(lambda x: '{}_{}'.format(x.q2_quantiles, x.q1_quantiles), axis=1))


def convert_to_onehot(x_raw, word2idx, max_len, pad_idx):
    encoded = []
    lengths = []
    for line in x_raw:
        words = re.split(r'\s+', line.strip())
        onehot = []
        for w in words:
            if w in word2idx:
                onehot.append(word2idx[w])
            else:
                j = hash(w) % 100
                onehot.append(word2idx[UNK + str(j)])

        padded = pad(onehot, max_len, pad_idx)
        encoded.append(padded)
        lengths.append(len(words))

    return encoded, lengths


def create_data_file(train_df):
    q1 = train_df.question1.values
    q2 = train_df.question2.values
    combined = np.concatenate((q1, q2))
    filename = os.path.join(os.path.dirname(__file__), 'data.txt')
    # noinspection PyTypeChecker
    np.savetxt(filename, combined, fmt='%s')


def load_bucketed_data(df):
    data = {}
    buckets = df.bucket.unique()
    for b in buckets:
        cases = df[df.bucket == b]
        data[b] = {
            'docs1': cases.qid1.values,
            'docs2': cases.qid2.values,
            'outcomes': (cases.is_duplicate == True).astype(int).values
        }

    data['docs1'] = df.set_index('qid1').to_dict()['q1_encoded']
    data['docs2'] = df.set_index('qid2').to_dict()['q2_encoded']

    return data


def load_embeddings(embed_filename):
    embeddings = []
    word2idx = {PAD: 0}
    for j in range(100):
        word2idx[UNK + str(j)] = j + 1

    input_dim, embed_dim = None, None
    with open(embed_filename, 'r') as f:
        for i, line in enumerate(f):
            values = line.split()
            if i == 0:  # header line that has dimensions
                input_dim, embed_dim = map(int, values)
                embeddings.append(np.random.normal(0, 1, embed_dim))  # __PAD__
                for j in range(100):
                    embeddings.append(np.random.normal(0, 1, embed_dim))  # __UNK__<j>
            else:
                word = values[0]
                word2idx[word] = i + 100  # PAD + 100 * UNK embeddings - header line
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings.append(coefs)

    return np.asarray(embeddings), word2idx, input_dim + 101, embed_dim


def load_question_pairs_dataset(test_size=1000):
    train_df = pd.read_csv(DATA_DIR + 'question_pairs/train_full.csv', header=0)
    train_df = train_df.astype({
        'id': int,
        'qid1': int,
        'qid2': int,
        'question1': str,
        'question2': str,
        'is_duplicate': int
    })
    # test_df = pd.read_csv(DATA_DIR + 'question_pairs/test.csv', header=0)
    # test_df = test_df.astype({'test_id': int, 'question1': str, 'question2': str})
    return (train_df[['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']],
            None)  # test_df[['question1', 'question2']][:test_size])


def pad(x_encoded, max_len, pad_idx):
    """Pre-pad (or truncate) sequence to max_len."""
    n = len(x_encoded)
    if n < max_len:
        padded = [pad_idx] * (max_len - n) + x_encoded
    else:
        padded = x_encoded[:max_len]

    return padded


def preprocess(train_df, word2idx, max_len):
    q1 = train_df.question1.values
    q2 = train_df.question2.values

    pad_idx = word2idx[PAD]
    q1_encoded, q1_lengths = convert_to_onehot(q1, word2idx, max_len, pad_idx)
    q2_encoded, q2_lengths = convert_to_onehot(q2, word2idx, max_len, pad_idx)

    return train_df.assign(q1_encoded=q1_encoded, q2_encoded=q2_encoded,
                           q1_length=q1_lengths, q2_length=q2_lengths)
