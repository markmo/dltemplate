from collections import Counter
from IPython.display import display, HTML
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
import numpy as np
import sys


def compute_test_accuracy(model, test_data, word_to_id, tag_to_id):
    test_words, test_tags = zip(*[zip(*sentence) for sentence in test_data])
    test_words = to_matrix(test_words, word_to_id)
    test_tags = to_matrix(test_tags, tag_to_id)

    # predict tag probabilities of shape [batch, time, n_tags]
    predicted_tag_probabilities = model.predict(test_words, verbose=1)
    predicted_tags = predicted_tag_probabilities.argmax(axis=-1)

    # compute accuracy excluding padding
    numerator = np.sum(np.logical_and((predicted_tags == test_tags), (test_words != 0)))
    denominator = np.sum(test_words != 0)

    return float(numerator) / denominator


def draw(sentence):
    words, tags = zip(*sentence)
    # noinspection PyTypeChecker
    display(HTML('<table><tr>{tags}</tr><tr>{words}</tr></table>'.format(
        words='<td>{}</td>'.format('</td><td>'.join(words)),
        tags='<td>{}</td>'.format('</td></td>'.join(tags))
    )))


def generate_batches(sentences, all_tags, word_to_id, tag_to_id, batch_size, max_len=None, pad=0):
    """
    The length of every batch depends on the maximum sentence length within the batch.

    Keras models have a `fit_generator` method that accepts a python generator
    yielding one batch at a time.

    :param sentences:
    :param all_tags:
    :param word_to_id:
    :param tag_to_id:
    :param batch_size:
    :param max_len:
    :param pad:
    :return:
    """
    assert isinstance(sentences, np.ndarray), 'Ensure sentences is a numpy array'
    while True:
        indices = np.random.permutation(np.arange(len(sentences)))
        for start in range(0, len(indices) - 1, batch_size):
            batch_indices = indices[start:(start + batch_size)]
            batch_words, batch_tags = [], []
            for sent in sentences[batch_indices]:
                words, tags = zip(*sent)
                batch_words.append(words)
                batch_tags.append(tags)

            batch_words = to_matrix(batch_words, word_to_id, max_len, pad)
            batch_tags = to_matrix(batch_tags, tag_to_id, max_len, pad)
            batch_tags_1hot = to_categorical(batch_tags, len(all_tags)).reshape(batch_tags.shape + (-1,))
            yield batch_words, batch_tags_1hot


def get_word_counts(data):
    word_counts = Counter()
    for sentence in data:
        words, tags = zip(*sentence)
        word_counts.update(words)

    all_words = ['#EOS#', '#UNK#'] + list(list(zip(*word_counts.most_common(10000)))[0])
    return all_words, word_counts


def to_matrix(names, token_to_id, max_len=None, pad=0, dtype='int32', time_major=False):
    """
    Converts a list of names into an RNN digestible matrix with padding added
    after the end.

    :param names:
    :param token_to_id:
    :param max_len:
    :param pad:
    :param dtype:
    :param time_major:
    :return:
    """
    max_len = max_len or max(map(len, names))
    matrix = np.empty([len(names), max_len], dtype)
    matrix.fill(pad)
    for i in range(len(names)):
        name_ix = list(map(token_to_id.__getitem__, names[i]))[:max_len]
        matrix[i, :len(name_ix)] = name_ix

    return matrix.T if time_major else matrix


def train(model, train_data, test_data, all_tags, word_to_id, tag_to_id, constants):
    batch_size = constants['batch_size']
    model.compile('adam', 'categorical_crossentropy')
    model.fit_generator(generate_batches(train_data, all_tags, word_to_id, tag_to_id, batch_size),
                        len(train_data) / batch_size,
                        callbacks=[EvaluateAccuracy(test_data, word_to_id, tag_to_id)],
                        epochs=constants['n_epochs'])


class EvaluateAccuracy(Callback):
    """
    The tricky part is not to count accuracy after sentence ends (on padding)
    and making sure we count all the validation data exactly once.

    Keras callbacks allow you to write a custom code to be ran once every epoch
    or every minibatch.
    """

    def __init__(self, test_data, word_to_id, tag_to_id):
        super().__init__()
        self._test_data = test_data
        self._word_to_id = word_to_id
        self._tag_to_id = tag_to_id

    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.flush()
        print('\nMeasuring validation accuracy...')
        acc = compute_test_accuracy(self.model, self._test_data, self._word_to_id, self._tag_to_id)
        print('\nValidation accuracy: %.5f\n' % acc)
        sys.stdout.flush()
