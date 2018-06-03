from common.util import is_number
import random
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


def batch_to_ids(sentences, word2id, max_len):
    """
    Prepares batches of ids.

    Sequences are padded to match the longest sequence in the batch.
    If it's longer than max_len, then max_len is used instead.

    :param sentences: list of original sequence strings
    :param word2id: (dict) a mapping from original symbols to ids
    :param max_len: (int) max len of sequences allowed
    :return: a list of lists of ids, a list of actual lengths
    """
    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
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


def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    """
    Generates pairs of equations and solutions.

    Each equation has a form of two integers joined by an operator.
    Each solution is an integer result of the operation.

    :param allowed_operators: list of strings
    :param dataset_size: (int) number of equations to be generated
    :param min_value: (int) min value of each operand
    :param max_value: (int) max value of each operand
    :return: list of tuples of strings (equation, solution)
    """
    sample = []
    for _ in range(dataset_size):
        equation = (str(random.randint(min_value, max_value)) +
                    random.choice(allowed_operators) +
                    str(random.randint(min_value, max_value)))
        solution = str(eval(equation))
        sample.append((equation, solution))

    return sample


def get_symbol_to_id_mappings():
    word2id = {symbol: i for i, symbol in enumerate('^$#+-1234567890')}
    id2word = {i: symbol for symbol, i in word2id.items()}

    return word2id, id2word


def ids_to_sentences(ids, id2word):
    """
    Converts a sequence of ids to a sequence of symbols.

    :param ids: list of indices for the padded sequence
    :param id2word: (dict) mapping from ids to original symbols
    :return: list of symbols
    """
    return [id2word[i] for i in ids]


def sentence_to_ids(sentence, word2id, padded_len):
    """
    Converts a sequence of symbols to a padded sequence of their ids.

    We will treat the original characters of the sequence and the end
    symbol as the valid part of the input. We will store the actual
    length of the sequence, which includes the end symbol, but does
    not include the padding symbols.

    :param sentence: (str) input/output sequence of symbols
    :param word2id: (dict) a mapping from original symbols to ids
    :param padded_len: (int) desired length of the sequence
    :return: tuple of (list of ids, actual length of sentence including end symbol)
    """
    pad_id = word2id['#']
    sent_ids = [pad_id for _ in range(padded_len)]
    i = 0
    for i in range(min(len(sentence), padded_len - 1)):
        sent_ids[i] = word2id[sentence[i]]

    sent_ids[i + 1] = word2id['$']
    sent_len = i + 2

    return sent_ids, sent_len


def train(sess, model, train_set, test_set, word2id, id2word, constants):
    sess.run(tf.global_variables_initializer())
    n_epochs = constants['n_epochs']
    batch_size = constants['batch_size']
    max_len = constants['max_len']
    n_step = int(len(train_set) / batch_size)

    invalid_number_prediction_counts, all_model_predictions, all_ground_truth = [], [], []

    print('Start training...\n')
    for epoch in range(n_epochs):
        random.shuffle(train_set)
        random.shuffle(test_set)
        print('Train: epoch', epoch + 1)
        for n_iter, (x_batch, y_batch) in enumerate(generate_batches(train_set, batch_size)):
            x_ids, x_sent_lens = batch_to_ids(x_batch, word2id, max_len)
            y_ids, y_sent_lens = batch_to_ids(y_batch, word2id, max_len)
            preds, loss = model.train_on_batch(sess, x_ids, x_sent_lens,
                                               y_ids, y_sent_lens,
                                               constants['learning_rate'],
                                               constants['dropout_keep_prob'])
            if n_iter % 200 == 0:
                print('Epoch: [%d/%d], step: [%d/%d], loss: %f' %
                      (epoch + 1, n_epochs, n_iter + 1, n_step, loss))

        x_sent, y_sent = next(generate_batches(test_set, batch_size))

        # prepare test data (x_sent and y_sent) for predicting
        # quality and computing value of the loss function
        x, x_sent_lens = batch_to_ids(x_sent, word2id, max_len)
        y, y_sent_lens = batch_to_ids(y_sent, word2id, max_len)

        preds, loss = model.predict_for_batch_with_loss(sess, x, x_sent_lens,
                                                        y, y_sent_lens)

        print('Test: epoch', epoch + 1, 'loss:', loss)
        for x_, y_, p in list(zip(x, y, preds))[:3]:
            print('X:', ''.join(ids_to_sentences(x_, id2word)))
            print('Y:', ''.join(ids_to_sentences(y_, id2word)))
            print('O:', ''.join(ids_to_sentences(p, id2word)))
            print('')

        model_predictions, ground_truth = [], []
        invalid_number_prediction_count = 0

        # For the whole test set calculate ground-truth values (as integers)
        # and prediction values (also as integers) to calculate metrics.
        # If generated by model number is not correct (e.g. '1-1'),
        # increase invalid_number_prediction_count and don't append this
        # and corresponding ground-truth value to the arrays.
        for x_batch, y_batch in generate_batches(test_set, batch_size):
            x_ids, x_sent_lens = batch_to_ids(x_batch, word2id, max_len)
            y_ids, y_sent_lens = batch_to_ids(y_batch, word2id, max_len)

            preds = model.predict_for_batch(sess, x_ids, x_sent_lens)

            for y, p in zip(y_ids, preds):
                y_sent = ''.join(ids_to_sentences(y, id2word))
                y_sent = y_sent[:y_sent.find('$')]
                p_sent = ''.join(ids_to_sentences(p, id2word))
                p_sent = p_sent[:p_sent.find('$')]
                if is_number(p_sent):
                    model_predictions.append(int(p_sent))
                    ground_truth.append(int(y_sent))
                else:
                    invalid_number_prediction_count += 1

        all_model_predictions.append(model_predictions)
        all_ground_truth.append(ground_truth)
        invalid_number_prediction_counts.append(invalid_number_prediction_count)

    print('\n...training finished')
    return all_ground_truth, all_model_predictions, invalid_number_prediction_counts
