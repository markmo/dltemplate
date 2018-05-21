from collections import defaultdict
from common.nlp_eval_util import nlp_metrics
import numpy as np
import tensorflow as tf


def batches_generator(batch_size, tokens, tags, tok2idx, tag2idx,
                      shuffle=True, allow_smaller_last_batch=True):
    """
    Generates padded batches of tokens and tags.

    Neural Networks are usually trained with batches. It means that weight
    updates of the network are based on several sequences at every single
    time. The tricky part is that all sequences within a batch need to have
    the same length. So we will pad them with a special <PAD> token. It is
    also a good practice to provide the RNN with sequence lengths, so it
    can skip computations for padding parts.

    :param batch_size:
    :param tokens:
    :param tags:
    :param tok2idx:
    :param tag2idx:
    :param shuffle:
    :param allow_smaller_last_batch:
    :return:
    """
    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        xs = []
        ys = []
        max_len_token = 0
        for idx in order[batch_start:batch_end]:
            xs.append(words2idxs(tokens[idx], tok2idx))
            ys.append(tags2idxs(tags[idx], tag2idx))
            max_len_token = max(max_len_token, len(tags[idx]))

        # insert the data into numpy nd-arrays filled with padding indices
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tok2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(xs[n])
            x[n, :utt_len] = xs[n]
            lengths[n] = utt_len
            y[n, :utt_len] = ys[n]

        yield x, y, lengths


def build_dict(tokens_or_tags, special_tokens):
    """

    :param tokens_or_tags: a list of lists of tokens or tags
    :param special_tokens: some special tokens
    :return: dict, a mapping from tokens (or tags) to indices, and
             list, a mapping from indices to tokens (or tags).
             The first special token is at index 0.
    """
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []

    i = 0
    for t in special_tokens:
        if t not in tok2idx:
            tok2idx[t] = i
            idx2tok.append(t)
            i += 1

    for ts in tokens_or_tags:
        for t in ts:
            if t not in tok2idx:
                tok2idx[t] = i
                idx2tok.append(t)
                i += 1

    return tok2idx, idx2tok


def words2idxs(tokens, tok2idx):
    return [tok2idx[word] for word in tokens]


def tags2idxs(tags, tag2idx):
    return [tag2idx[tag] for tag in tags]


def idxs2words(idxs, idx2tok):
    return [idx2tok[idx] for idx in idxs]


def idx2tags(idxs, idx2tag):
    return [idx2tag[idx] for idx in idxs]


def predict_tags(model, sess, token_idxs_batch, lengths, idx2tok, idx2tag):
    """
    Performs predictions and transforms indices to tokens and tags.

    :param model:
    :param sess:
    :param token_idxs_batch:
    :param lengths:
    :param idx2tok:
    :param idx2tag:
    :return:
    """
    tag_idxs_batch = model.predict_for_batch(sess, token_idxs_batch, lengths)
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2tok[token_idx])

        tags_batch.append(tags)
        tokens_batch.append(tokens)

    return tags_batch, tokens_batch


PAD_TOKEN = '<PAD>'


def eval_conll(model, sess, tokens, tags, tok2idx, tag2idx, idx2tok, idx2tag, short_report=True):
    """
    Computes NER quality measures using CONLL shared task script.

    :param model:
    :param sess:
    :param tokens:
    :param tags:
    :param tok2idx:
    :param tag2idx:
    :param idx2tok:
    :param idx2tag:
    :param short_report:
    :return:
    """
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags, tok2idx, tag2idx):
        tags_batch, tokens_batch = predict_tags(model, sess, x_batch, lengths, idx2tok, idx2tag)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception('Incorrect length of prediction for the input, '
                            'expected length: %i, got: %i' % (len(x_batch[0]), len(tags_batch[0])))

        predicted_tags = []
        ground_truth_tags = []
        for true_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]):
            if token != PAD_TOKEN:
                ground_truth_tags.append(idx2tag[true_tag_idx])
                predicted_tags.append(pred_tag)

        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])

    return nlp_metrics(y_true, y_pred, print_results=True, short_report=short_report)


def train(model, sess, tokens_train, tags_train, tokens_val, tags_val, tok2idx, tag2idx, idx2tok, idx2tag, constants):
    sess.run(tf.global_variables_initializer())
    n_epochs = constants['n_epochs']
    batch_size = constants['batch_size']
    learning_rate = constants['learning_rate']
    learning_rate_decay = constants['learning_rate_decay']
    dropout_keep_prob = constants['dropout_keep_prob']

    print('n_epochs:', n_epochs)
    print('batch_size:', batch_size)
    print('learning_rate:', learning_rate)
    print('learning_rate_decay:', learning_rate_decay)
    print('dropout_keep_prob:', dropout_keep_prob)
    print('Start training...\n')

    for epoch in range(n_epochs):
        # For each epoch evaluate the model on train and validation data
        print('-' * 20 + ' Epoch {} of {} '.format(epoch + 1, n_epochs) + '-' * 20)
        print('Training data evaluation:')
        eval_conll(model, sess, tokens_train, tags_train, tok2idx, tag2idx, idx2tok, idx2tag, short_report=True)

        print('Validation data evaluation:')
        eval_conll(model, sess, tokens_val, tags_val, tok2idx, tag2idx, idx2tok, idx2tag, short_report=True)

        # Train the model
        generator = batches_generator(batch_size, tokens_train, tags_train, tok2idx, tag2idx)
        for x_batch, y_batch, lengths in generator:
            model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_prob)

        # Decay the learning rate
        learning_rate = learning_rate / learning_rate_decay

    print('...Training done')
