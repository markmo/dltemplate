from collections import Counter
import itertools
import logging
import numpy as np
import re

logging.getLogger().setLevel(logging.INFO)


def batch_iter(data, batch_size, n_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    n_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(n_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(n_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
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


def load_embeddings(vocab):
    embeddings = {}
    for word in vocab:
        embeddings[word] = np.random.uniform(-0.25, 0.25, 300)

    return embeddings


def pad_sentences(sentences, pad_token='<PAD>', forced_seq_len=None):
    if forced_seq_len is None:  # Train
        seq_len = max(len(sent) for sent in sentences)
    else:  # Prediction
        logging.critical('In prediction, reading trained sequence length...')
        seq_len = forced_seq_len

    logging.critical('Max sequence length: {}'.format(seq_len))
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


def train():
    pass
