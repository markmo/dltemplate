from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


START_SYMBOL = '<S>'
END_SYMBOL = '<E>'
PAD_SYMBOL = '<P>'
UNK_SYMBOL = '<U>'


def batch_to_ids(sentences, embeddings, word2id, max_len):
    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, embeddings, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)

    return batch_ids, batch_ids_len


def evaluate_results(all_ground_truth, all_model_predictions, invalid_number_prediction_counts):
    collection = zip(all_ground_truth, all_model_predictions, invalid_number_prediction_counts)
    for i, (gts, preds, n_invalid) in enumerate(collection, 1):
        err = mean_absolute_error(gts, preds)
        print('Epoch: %i, MAE: %f, Invalid numbers: %i' % (i, err, n_invalid))


def generate_batches(samples, batch_size=64):
    x, y = [], []
    for i, (x_, y_) in enumerate(samples, 1):
        x.append(x_)
        y.append(y_)
        if i % batch_size == 0:
            yield x, y
            x, y = [], []

    if x and y:
        yield x, y


def get_symbol_to_id_mappings(vocab):
    word2id = {symbol: i for i, symbol in enumerate(vocab)}
    id2word = {i: symbol for symbol, i in word2id.items()}

    return word2id, id2word


def get_word_embeddings(embeddings, id2word, vocab_size, embeddings_size=300):
    word_embeddings = np.zeros((vocab_size, embeddings_size), dtype='float32')
    word_embeddings[0, :] = 0.
    word_embeddings[1, :] = 1.
    word_embeddings[2, :] = -1.
    word_embeddings[3, :] = 0.
    for i in range(4, vocab_size):
        word_embeddings[i, :] = embeddings[id2word[i]]

    return word_embeddings


def get_vocab(data):
    """

    :param data: list of (utterance, reply)
    :return: dict of word_counts, (vocab) list of words, vocab size
    """
    tokens = defaultdict(int)
    for u, r in data:
        for t in u + r:
            tokens[t] += 1

    vocab = [START_SYMBOL, END_SYMBOL, PAD_SYMBOL, UNK_SYMBOL]
    vocab.extend(tokens.keys())

    return tokens, vocab, len(vocab)


def ids_to_sentences(ids, id2word):
    """
    Converts a sequence of ids to a sequence of symbols.

    :param ids: list of indices for the padded sequence
    :param id2word: (dict) mapping from ids to original symbols
    :return: list of symbols
    """
    return [id2word[i] for i in ids]


def sentence_to_ids(sentence, embeddings, word2id, padded_len):
    pad_id = word2id[PAD_SYMBOL]
    unk_id = word2id[UNK_SYMBOL]
    sent_ids = [pad_id for _ in range(padded_len)]
    i = 0
    for i in range(min(len(sentence), padded_len - 1)):
        word = sentence[i]
        if word not in embeddings:
            sent_ids[i] = unk_id
        else:
            sent_ids[i] = word2id[sentence[i]]

    sent_ids[i + 1] = word2id[END_SYMBOL]
    sent_len = i + 2

    return sent_ids, sent_len


def train(sess, model, train_set, test_set, embeddings, word2id, id2word,
          n_epochs, batch_size, max_len, learning_rate, dropout_keep_prob):
    sess.run(tf.global_variables_initializer())
    n_step = int(len(train_set) / batch_size)

    invalid_number_prediction_counts, all_model_predictions, all_ground_truth = [], [], []

    print('Start training...\n')
    for epoch in range(n_epochs):
        random.shuffle(train_set)
        random.shuffle(test_set)
        print('Train: epoch', epoch + 1)
        for n_iter, (x_batch, y_batch) in enumerate(generate_batches(train_set, batch_size)):
            print('batch:', n_iter)
            x_ids, x_sent_lens = batch_to_ids(x_batch, embeddings, word2id, max_len)
            y_ids, y_sent_lens = batch_to_ids(y_batch, embeddings, word2id, max_len)
            preds, loss = model.train_on_batch(sess, x_ids, x_sent_lens,
                                               y_ids, y_sent_lens,
                                               learning_rate,
                                               dropout_keep_prob)
            if n_iter % 200 == 0:
                print('Epoch: [%d/%d], step: [%d/%d], loss: %f' %
                      (epoch + 1, n_epochs, n_iter + 1, n_step, loss))

        x_sent, y_sent = next(generate_batches(test_set, batch_size))

        # prepare test data (x_sent and y_sent) for predicting
        # quality and computing value of the loss function
        x, x_sent_lens = batch_to_ids(x_sent, embeddings, word2id, max_len)
        y, y_sent_lens = batch_to_ids(y_sent, embeddings, word2id, max_len)

        preds, loss = model.predict_for_batch_with_loss(sess, x, x_sent_lens,
                                                        y, y_sent_lens)

        print('Test: epoch', epoch + 1, 'loss:', loss)
        for x_, y_, p in list(zip(x, y, preds))[:3]:
            print('X:', ' '.join(ids_to_sentences(x_, id2word)))
            print('Y:', ' '.join(ids_to_sentences(y_, id2word)))
            print('O:', ' '.join(ids_to_sentences(p, id2word)))
            print('')

        model_predictions, ground_truth = [], []
        invalid_number_prediction_count = 0

        # For the whole test set calculate ground-truth values (as integers)
        # and prediction values (also as integers) to calculate metrics.
        # If generated by model number is not correct (e.g. '1-1'),
        # increase invalid_number_prediction_count and don't append this
        # and corresponding ground-truth value to the arrays.
        for x_batch, y_batch in generate_batches(test_set, batch_size):
            x_ids, x_sent_lens = batch_to_ids(x_batch, embeddings, word2id, max_len)
            y_ids, y_sent_lens = batch_to_ids(y_batch, embeddings, word2id, max_len)

            preds = model.predict_for_batch(sess, x_ids, x_sent_lens, y_sent_lens)

            for y, p in zip(y_ids, preds):
                y_sent = ' '.join(ids_to_sentences(y, id2word))
                y_sent = y_sent[:y_sent.find(END_SYMBOL)]
                p_sent = ' '.join(ids_to_sentences(p, id2word))
                p_sent = p_sent[:p_sent.find(END_SYMBOL)]
                model_predictions.append(p_sent)
                ground_truth.append(y_sent)

        all_model_predictions.append(model_predictions)
        all_ground_truth.append(ground_truth)
        invalid_number_prediction_counts.append(invalid_number_prediction_count)

    print('\n...training finished')
    return all_ground_truth, all_model_predictions, invalid_number_prediction_counts
