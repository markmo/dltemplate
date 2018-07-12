from common.load_data import load_hebrew_to_english_dataset
from functools import partial
import numpy as np
import pytest
import random
from rl.rl_for_seq2seq.util import compute_levenshtein, get_distance, Vocab
from sklearn.model_selection import train_test_split
import tensorflow as tf


@pytest.fixture
def data():
    word_to_translation = load_hebrew_to_english_dataset()
    all_words = np.array(list(word_to_translation.keys()))
    all_translations = np.array([ts for all_ts in word_to_translation.values() for ts in all_ts])
    train_words, test_words = train_test_split(all_words, test_size=0.1, random_state=42)
    bos = '_'
    eos = ';'
    inp_voc = Vocab.from_lines(''.join(all_words), bos=bos, eos=eos, sep='')
    out_voc = Vocab.from_lines(''.join(all_translations), bos=bos, eos=eos, sep='')
    return train_words, inp_voc, out_voc, word_to_translation, all_translations


@pytest.fixture
def sess():
    tf.reset_default_graph()
    return tf.InteractiveSession()


# noinspection PyShadowingNames,PyUnusedLocal
def test_compute_levenshtein(data, sess):
    train_words, inp_voc, out_voc, word_to_translation, all_translations = data
    batch_words = np.random.choice(train_words, size=100)
    batch_trans = list(map(random.choice, map(word_to_translation.get, batch_words)))
    batch_trans_wrong = np.random.choice(all_translations, size=100)
    batch_words_ix = tf.constant(inp_voc.to_matrix(batch_words))
    batch_trans_ix = tf.constant(out_voc.to_matrix(batch_trans))
    batch_trans_wrong_ix = tf.constant(out_voc.to_matrix(batch_trans_wrong))

    correct_answers_score = compute_levenshtein(batch_words_ix, batch_trans_ix,
                                                inp_voc, out_voc, word_to_translation).eval()
    assert np.all(correct_answers_score == 0)

    wrong_answers_score = compute_levenshtein(batch_words_ix, batch_trans_wrong_ix,
                                              inp_voc, out_voc, word_to_translation).eval()
    get_distance_partial = partial(get_distance, word_to_translation=word_to_translation)
    true_wrong_answers_score = np.array(list(map(get_distance_partial, batch_words, batch_trans_wrong)))

    assert np.all(wrong_answers_score == true_wrong_answers_score)
