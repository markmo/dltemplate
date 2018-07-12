import editdistance
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tqdm import trange


class Vocab(object):

    def __init__(self, tokens, bos='__BOS__', eos='__EOS__', sep=''):
        """ handles tokenizing and detokenizing """
        assert bos in tokens, eos in tokens
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.bos = bos
        self.bos_id = self.token_to_id[bos]
        self.eos = eos
        self.eos_id = self.token_to_id[eos]
        self.sep = sep

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, bos='__BOS__', eos='__EOS__', sep=''):
        flat_lines = sep.join(list(lines))
        flat_lines = list(flat_lines.split(sep)) if sep != '' else list(flat_lines)
        tokens = list(set(sep.join(flat_lines)))
        tokens = [t for t in tokens if t not in (bos, eos) and len(t) != 0]
        tokens = [bos, eos] + tokens
        return Vocab(tokens, bos, eos, sep)

    def tokenize(self, string):
        """ converts string to a list of tokens """
        tokens = list(filter(len, string.split(self.sep))) if self.sep != '' else list(string)
        return [self.bos] + tokens + [self.eos]

    def to_matrix(self, lines, max_len=None):
        """
        Convert variable length token sequences into a fixed size matrix.

        Example usage:

        `print(as_matrix(words[:3], source_to_id))
        [[15 22 21 28 27 13 -1 -1 -1 -1 -1]
         [30 21 15 15 21 14 28 27 13 -1 -1]
         [25 37 31 34 21 20 37 21 28 19 13]]`

        :param lines:
        :param max_len:
        :return:
        """
        max_len = max_len or max(map(len, lines)) + 2  # for bos and eos
        matrix = np.zeros((len(lines), max_len), dtype='int32') + self.eos_id
        for i, seq in enumerate(lines):
            tokens = self.tokenize(seq)
            row_id = list(map(self.token_to_id.get, tokens))[:max_len]
            matrix[i, :len(row_id)] = row_id

        return matrix

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings

        :param matrix: matrix of tokens of int32, shape=[batch, time]
        :param crop: if True, crops BOS and EOS from line
        :return:
        """
        lines = []
        for indices in map(list, matrix):
            if crop:
                if indices[0] == self.bos_id:
                    indices = indices[1:]

                if self.eos_id in indices:
                    indices = indices[:indices.index(self.eos_id)]

            line = self.sep.join(self.tokens[ix] for ix in indices)
            lines.append(line)

        return lines


def initialize_uninitialized(sess=None):
    """
    Initialize uninitialized variables, doesn't affect those already initialized

    :param sess: session to initialize within (defaults to `tf.get_default_session`
    :return:
    """
    sess = sess or tf.get_default_session()
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    uninitialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(uninitialized_vars):
        sess.run(tf.variables_initializer(uninitialized_vars))


def infer_length(seq, eos_id, time_major=False, dtype=tf.int32):
    """
    Compute length given output indices and EOS id.

    :param seq: tf matrix [time, batch] if time_major else [batch, time]
    :param eos_id: (int) index of EOS id
    :param time_major:
    :param dtype:
    :return: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_id), dtype)
    count_eos = tf.cumsum(is_eos, axis=axis, exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), dtype), axis=axis)
    return lengths


def infer_mask(seq, eos_id, time_major=False, dtype=tf.float32):
    """
    Compute mask given output indices and EOS id.

    :param seq: tf matrix [time, batch] if time_major else [batch, time]
    :param eos_id: (int) index of EOS id
    :param time_major:
    :param dtype:
    :return: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_id, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major:
        mask = tf.transpose(mask)

    return mask


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.

    :param values: logits for all actions, float32[batch, tick, action]
    :param indices: action ids, int32[batch, tick]
    :return: values selected for the given actions, float[batch, tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0, batch_size)[:, None], [1, seq_len])
    time_i = tf.tile(tf.range(0, seq_len)[None, :], [batch_size, 1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values, indices_nd)


def get_distance(word, trans, word_to_translation):
    """
    A function that takes word and predicted translation and evaluates
    (Levenshtein's) edit distance to closest correct translation.

    :param word:
    :param trans:
    :param word_to_translation:
    :return:
    """
    references = word_to_translation[word]
    assert len(references) != 0, 'wrong/unknown word'
    return min(editdistance.eval(trans, ref) for ref in references)


def score(words, word_to_translation, translate_fn, batch_size=100):
    """ a function that computes levenshtein distance for batch_size random samples """
    assert isinstance(words, np.ndarray)
    batch_words = np.random.choice(words, size=batch_size, replace=False)
    batch_trans = translate_fn(batch_words)
    get_distance_partial = partial(get_distance, word_to_translation=word_to_translation)
    distances = list(map(get_distance_partial, batch_words, batch_trans))
    return np.array(distances, dtype='float32')


class SupervisedTrainer(object):
    """
    Trains model through maximizing log-likelihood ~ minimizing crossentropy.
    """
    def __init__(self, model, out_voc):
        # variables for inputs and correct answers
        self.input_sequence = tf.placeholder('int32', [None, None])
        self.reference_answers = tf.placeholder('int32', [None, None])

        # Compute log-probabilities of all possible tokens at each step. Use model interface.
        log_probs_seq = model.symbolic_score(self.input_sequence, self.reference_answers)

        # Compute mean crossentropy
        crossentropy = -select_values_over_last_axis(log_probs_seq, self.reference_answers)

        mask = infer_mask(self.reference_answers, out_voc.eos_id)
        self.loss = tf.reduce_sum(crossentropy * mask) / tf.reduce_sum(mask)

        # Build weights optimizer. Use `model.weights` to get all trainable params.
        self.train_step = model.weights


def sample_batch(words, word_to_translation, inp_voc, out_voc, batch_size):
    """
    Sample random batch of words and random correct translation for each word.

    Example usage:

        batch_x, batch_y = sample_batch(train_words, word_to_translations, 10)

    :param words:
    :param word_to_translation:
    :param inp_voc:
    :param out_voc:
    :param batch_size:
    :return:
    """
    # choose words
    batch_words = np.random.choice(words, size=batch_size)

    # choose translations
    batch_trans_candidates = list(map(word_to_translation.get, batch_words))
    batch_trans = list(map(random.choice, batch_trans_candidates))

    return inp_voc.to_matrix(batch_words), out_voc.to_matrix(batch_trans)


def train(train_words, test_words, word_to_translation, inp_voc, out_voc,
          translate_fn, sup_trainer, sess, n_epochs=25000, report_freq=100):
    loss_history = []
    editdist_history = []
    current_scores = []
    for i in trange(n_epochs):
        bx, by = sample_batch(train_words, word_to_translation, inp_voc, out_voc, batch_size=32)
        feed_dict = {
            sup_trainer.input_sequence: bx,
            sup_trainer.reference_answers: by
        }
        loss, _ = sess.run([sup_trainer.loss, sup_trainer.train_step], feed_dict)
        loss_history.append(loss)
        if (i + 1) % 100 == 0:
            current_scores = score(test_words, word_to_translation, translate_fn)
            editdist_history.append(current_scores.mean())

        if (i + 1) % report_freq == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.title('train loss / training time')
            plt.plot(loss_history)
            plt.grid()
            plt.subplot(132)
            plt.title('val score distribution')
            plt.hist(current_scores, bins=20)
            plt.subplot(133)
            plt.title('val score / training time')
            plt.plot(editdist_history)
            plt.grid()
            plt.show()
            mean_loss: float = np.mean(loss_history[-10:])
            mean_score: float = np.mean(editdist_history[-10:])
            print('llh=%.3f, mean score=%.3f' % (mean_loss, mean_score))


def _compute_levenshtein(words_ix, trans_ix, inp_voc, out_voc, word_to_translation):
    """

    :param words_ix:
    :param trans_ix:
    :param inp_voc:
    :param out_voc:
    :param word_to_translation:
    :return:
    """
    words = inp_voc.to_lines(words_ix)

    assert type(words) is list and type(words[0]) is str and len(words) == len(words_ix)

    trans = out_voc.to_lines(trans_ix)

    assert type(trans) is list and type(trans[0]) is str and len(trans) == len(trans_ix)

    distances = [get_distance(words[i], trans[i], word_to_translation) for i in range(len(words))]

    assert type(distances) in (list, tuple, np.ndarray) and len(distances) == len(words_ix)

    distances = np.array(list(distances), dtype='float32')

    return distances


def compute_levenshtein(words_ix, trans_ix, inp_voc, out_voc, word_to_translation):
    """ A custom tensorflow operation that computes Levenshtein loss for predicted trans. """
    partial_fn = partial(_compute_levenshtein,
                         inp_voc=inp_voc,
                         out_voc=out_voc,
                         word_to_translation=word_to_translation)
    out = tf.py_func(partial_fn, [words_ix, trans_ix], tf.float32)
    out.set_shape([None])

    return tf.stop_gradient(out)


class Trainer(object):
    """
    Implements an algorithm called self-critical sequence training.

    See https://arxiv.org/abs/1612.00563.

    The algorithm is a vanilla policy gradient with a special baseline.

        delta_J = Ex~p(s) Ey~pi(y|x) delta_log pi(y|x) . (R(x, y) - b(x))

    Here reward R(x,y) is a negative Levenshtein distance (since we minimize it).
    The baseline b(x) represents how well the model fares on word x.

    In practice, this means that we compute baseline as a score of greedy translation,

        b(x)=R(x, y_greedy(x)).

    """
    def __init__(self, model, inp_voc, out_voc, word_to_translation):
        self.input_sequence = tf.placeholder('int32', [None, None])

        # use model to __sample__ symbolic translations given input_sequence
        sample_translations, sample_logp = model.symbolic_translate(self.input_sequence, greedy=False)

        # use model to get __greedy__ symbolic translations given input_sequence
        greedy_translations, greedy_logp = model.symbolic_translate(self.input_sequence, greedy=True)

        rewards = -compute_levenshtein(self.input_sequence, sample_translations, inp_voc, out_voc, word_to_translation)

        # compute __negative__ Levenshtein distance in greedy mode
        baseline = -compute_levenshtein(self.input_sequence, greedy_translations, inp_voc, out_voc, word_to_translation)

        # compute advantage using rewards and baseline
        advantage = rewards - baseline
        assert advantage.shape.ndims == 1, 'advantage must be of shape [batch_size]'

        # compute log_pi(a_t|s_t), shape=[batch, seq_length]
        logprobs_phoneme = select_values_over_last_axis(sample_logp, sample_translations)

        # compute policy gradient, or rather, surrogate function whose gradient is the policy gradient
        j = logprobs_phoneme * advantage[:, None]

        mask = infer_mask(sample_translations, out_voc.eos_id)
        self.loss = -tf.reduce_sum(j * mask) / tf.reduce_sum(mask)

        # regularize using negative entropy
        # note: for entropy you need probabilities for all tokens (sample_logp),
        # not just logprobs_phoneme
        # entropy matrix of shape [batch, seq_length]:
        entropy = -tf.reduce_sum(tf.exp(sample_logp) * sample_logp, axis=-1)

        assert entropy.shape.ndims == 2, 'make sure elementwise entropy is shape [batch, time]'

        self.loss -= 0.01 * tf.reduce_sum(entropy * mask) / tf.reduce_sum(mask)

        # compute weight updates, clip by norm
        grads = tf.gradients(self.loss, model.weights)
        grads = tf.clip_by_global_norm(grads, 50)[0]

        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-5).apply_gradients(zip(grads, model.weights))


def train_policy_gradients(train_words, test_words, word_to_translation, inp_voc, out_voc,
                           translate_fn, trainer, sess, n_epochs=100000, batch_size=32, report_freq=100):
    loss_history = []
    editdist_history = []
    current_scores = []
    for i in trange(n_epochs):
        bx = sample_batch(train_words, word_to_translation, inp_voc, out_voc, batch_size)[0]
        pseudo_loss, _ = sess.run([trainer.loss, trainer.train_step], {trainer.input_sequence: bx})
        loss_history.append(pseudo_loss)
        if (i + 1) % 100 == 0:
            current_scores = score(test_words, word_to_translation, translate_fn)
            editdist_history.append(current_scores.mean())

        if (i + 1) % report_freq == 0:
            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            plt.title('val score distribution')
            plt.hist(current_scores, bins=20)
            plt.subplot(122)
            plt.title('val score / training time')
            plt.plot(editdist_history)
            plt.grid()
            plt.show()
            mean_loss: float = np.mean(loss_history[-10:])
            mean_score: float = np.mean(editdist_history[-10:])
            print('J=%.3f, mean score=%.3f' % (mean_loss, mean_score))


def evaluate(train_words, test_words, word_to_translation, translate_fn):
    """
    If you get 'Out Of Memory', replace this with batched computation

    :param train_words:
    :param test_words:
    :param word_to_translation:
    :param translate_fn:
    :return:
    """
    for word in train_words[:10]:
        print('%s -> %s' % (word, translate_fn([word])[0]))

    test_scores = []
    for start_i in trange(0, len(test_words), 32):
        batch_words = test_words[start_i:start_i+32]
        batch_trans = translate_fn(batch_words)
        get_distance_partial = partial(get_distance, word_to_translation=word_to_translation)
        distances = list(map(get_distance_partial, batch_words, batch_trans))
        test_scores.extend(distances)

    print('Supervised test score:', np.mean(test_scores))
