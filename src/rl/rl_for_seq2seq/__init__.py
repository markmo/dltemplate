from argparse import ArgumentParser
from common.load_data import load_hebrew_to_english_dataset
from common.util import load_hyperparams, merge_dict
import matplotlib.pyplot as plt
import numpy as np
import os
from rl.rl_for_seq2seq.model_setup import BasicTranslationModel
from rl.rl_for_seq2seq.util import initialize_uninitialized, get_distance, evaluate
from rl.rl_for_seq2seq.util import SupervisedTrainer, train, Trainer, train_policy_gradients, Vocab
from sklearn.model_selection import train_test_split
import tensorflow as tf


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    mode = constants['mode']
    easy_mode = constants['easy_mode']
    # max_output_length = 20 if easy_mode else 50

    word_to_translation = load_hebrew_to_english_dataset(mode, easy_mode)
    all_words = np.array(list(word_to_translation.keys()))
    all_translations = np.array([ts for all_ts in word_to_translation.values() for ts in all_ts])

    # split the dataset
    train_words, test_words = train_test_split(all_words, test_size=0.1, random_state=42)

    # build vocab
    bos = '_'
    eos = ';'
    inp_voc = Vocab.from_lines(''.join(all_words), bos=bos, eos=eos, sep='')
    out_voc = Vocab.from_lines(''.join(all_translations), bos=bos, eos=eos, sep='')

    # test casting lines into ids and back again
    batch_lines = all_words[:5]
    batch_ids = inp_voc.to_matrix(batch_lines)
    batch_lines_restored = inp_voc.to_lines(batch_ids)
    print('lines:')
    print(batch_lines)
    print('\nwords to ids (0=bos, 1=eos):')
    print(batch_ids)
    print('\nback to words:')
    print(batch_lines_restored)

    # plot word/translation length distributions to estimate the scope of the task
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.title('words')
    plt.hist(list(map(len, all_words)), bins=20)

    plt.subplot(1, 2, 2)
    plt.title('translations')
    plt.hist(list(map(len, all_translations)), bins=20)

    model = BasicTranslationModel('model', inp_voc, out_voc, n_embedding=64, n_hidden=128)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # test translation
    input_sequence = tf.placeholder('int32', [None, None])
    greedy_translations, logp = model.symbolic_translate(input_sequence, greedy=True)

    def translate(lines):
        """
        You are given a list of input lines,
        make your neural network translate them.

        :param lines:
        :return: a list of output lines
        """
        # convert lines to a matrix of indices
        lines_ix = inp_voc.to_matrix(lines)

        # compute translations in form of indices
        trans_ix = sess.run(greedy_translations, {input_sequence: lines_ix})

        # convert translations back into strings
        return out_voc.to_lines(trans_ix)

    print('Sample inputs:', all_words[:3])
    print('Dummy translations:', translate(all_words[:3]))

    assert isinstance(greedy_translations, tf.Tensor) and greedy_translations.dtype.is_integer, \
        'translation must be a tensor of integers (token ids)'
    assert translate(all_words[:3]) == translate(all_words[:3]), \
        'make sure translation is deterministic (use greedy=True and disable any noise layers)'
    # assert type(translate(all_words[:3])) is list and \
    #        (type(translate(all_words[:1])[0]) is str or type(translate(all_words[:1])[0]) is unicode), \
    #     'translate(lines) must return a sequence of strings'
    assert type(translate(all_words[:3])) is list and type(translate(all_words[:1])[0]) is str, \
        'translate(lines) must return a sequence of strings'
    print('Tests passed!')

    # initialize optimizer params while keeping model intact
    initialize_uninitialized(sess)

    n_epochs = constants['n_epochs']
    # report_freq = constants['report_freq']

    sup_trainer = SupervisedTrainer(model, out_voc)
    train(train_words, test_words, word_to_translation, inp_voc, out_voc,
          translate, sup_trainer, sess, n_epochs, report_freq=5000)

    evaluate(train_words, test_words, word_to_translation, translate)

    # Self-critical policy gradient
    trainer = Trainer(model, inp_voc, out_voc, word_to_translation)

    initialize_uninitialized(sess)

    train_policy_gradients(train_words, test_words, word_to_translation, inp_voc, out_voc,
                           translate, trainer, sess, n_epochs=100000, batch_size=32, report_freq=20000)

    evaluate(train_words, test_words, word_to_translation, translate)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Hebrew-to-English translation model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    args = parser.parse_args()

    run(vars(args))
