import json
import logging
import os
import pyrouge
import tensorflow as tf
from tf_model.pointer_generator.beam_search import run_beam_search
from tf_model.pointer_generator.data import output_ids2words, show_abs_oovs, show_art_oovs, STOP_DECODING
from tf_model.pointer_generator.util import get_config, load_ckpt
import time


SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


# noinspection SpellCheckingInspection
class BeamSearchDecoder(object):

    def __init__(self, model, batcher, vocab, constants):
        """

        :param model: Seq2SeqAttentionModel object
        :param batcher: Batcher object
        :param vocab: Vocabulary object
        :param constants:
        """
        self._model = model
        model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._min_dec_steps = constants['min_dec_steps']
        self._max_dec_steps = constants['max_dec_steps']
        self._beam_size = constants['beam_size']
        self._single_pass = constants['single_pass']
        self._pointer_gen = constants['pointer_gen']
        self._log_root = constants['log_root']
        self._saver = tf.train.Saver()  # we use this to load checkpoints for decoding
        self._sess = tf.Session(config=get_config())

        # Load an initial checkpoint to use for decoding
        ckpt_path = load_ckpt(self._saver, self._sess, self._log_root)

        if self._single_pass:
            # Make a descriptive decode directory name
            ckpt_name = 'ckpt-' + ckpt_path.split('-')[-1]  # this is something of the form "ckpt-123456"
            self._decode_dir = os.path.join(self._log_root, get_decode_dir_name(ckpt_name,
                                                                                constants['data_path'],
                                                                                constants['max_enc_steps'],
                                                                                self._min_dec_steps,
                                                                                self._max_dec_steps,
                                                                                self._beam_size))
            if os.path.exists(self._decode_dir):
                raise Exception('single_pass decode directory %s should not already exist' % self._decode_dir)

        else:  # Generic decode dir name
            self._decode_dir = os.path.join(constants['log_root'], 'decode')

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir):
            os.mkdir(self._decode_dir)

        if self._single_pass:
            # Make the dirs to contain output written in the correct format for pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, 'reference')
            if not os.path.exists(self._rouge_ref_dir):
                os.mkdir(self._rouge_ref_dir)

            self._rouge_dec_dir = os.path.join(self._decode_dir, 'decoded')
            if not os.path.exists(self._rouge_dec_dir):
                os.mkdir(self._rouge_dec_dir)

    def decode(self):
        """
        Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode
        indefinitely, loading latest checkpoint at regular intervals.

        :return:
        """
        t0 = time.time()
        counter = 0
        while True:
            batch = self._batcher.next_batch()  # one example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                assert self._single_pass, 'Dataset exhausted, but we are not in single_pass mode'
                tf.logging.info('Decoder has finished reading dataset for single_pass.')
                tf.logging.info('Output has been saved in %s and %s. Now starting ROUGE eval...',
                                self._rouge_ref_dir, self._rouge_dec_dir)
                results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_with_unks = show_art_oovs(original_article, self._vocab)  # string
            abstract_with_unks = show_abs_oovs(original_abstract, self._vocab,
                                               (batch.art_oovs[0] if self._pointer_gen else None))  # string

            # Run beam search to get best Hypothesis
            best_hyp = run_beam_search(self._sess, self._model, self._vocab, batch,
                                       self._min_dec_steps, self._max_dec_steps, self._beam_size)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = output_ids2words(output_ids, self._vocab,
                                             (batch.art_oovs[0] if self._pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            decoded_output = ' '.join(decoded_words)  # string

            if self._single_pass:
                # write ref summary and decoded summary to file, to eval with pyrouge later
                self.write_for_rouge(original_abstract_sents, decoded_words, counter)
                counter += 1  # this is how many examples we've decoded
            else:
                # log output to screen
                print_results(article_with_unks, abstract_with_unks, decoded_output)

                # write info to .json file for visualization tool
                self.write_for_attnvis(article_with_unks, abstract_with_unks, decoded_words,
                                       best_hyp.attn_dists, best_hyp.p_gens)

                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
                t1 = time.time()
                if t1 - t0 > SECS_UNTIL_NEW_CKPT:
                    tf.logging.info("We've been decoding with same checkpoint for %i seconds. " +
                                    "Time to load new checkpoint", t1 - t0)
                    _ = load_ckpt(self._saver, self._sess, self._log_root)
                    t0 = time.time()

    def write_for_rouge(self, reference_sents, decoded_words, ex_index):
        """
        Write output to file in correct format for eval with pyrouge.
        This is called in single_pass mode.

        :param reference_sents: list of strings
        :param decoded_words: list of strings
        :param ex_index: (int) index with which to label the files
        :return:
        """
        # First, divide decoded output into sentences
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index('.')
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)

            sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        # Write to file
        ref_file = os.path.join(self._rouge_ref_dir, '%06d_reference.txt' % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, '%06d_decoded.txt' % ex_index)

        with open(ref_file, 'w') as f:
            for i, sent in enumerate(reference_sents):
                f.write(sent) if i == len(reference_sents) - 1 else f.write(sent + '\n')

        with open(decoded_file, 'w') as f:
            for i, sent in enumerate(decoded_sents):
                f.write(sent) if i == len(decoded_sents) - 1 else f.write(sent + '\n')

        tf.logging.info('Wrote example %i to file' % ex_index)

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        """
        Write some data to json file, which can be read into the in-browser
        attention visualizer tool:

        https://github.com/abisee/attn_vis

        :param article: original article string
        :param abstract: human (correct) abstract string
        :param decoded_words: list of arrays; the attention distributions
        :param attn_dists: list of strings; the words of the generated summary
        :param p_gens: list of scalars; the p_gen values. If not running in
               pointer-generator mode, list of None.
        :return:
        """
        article_lst = article.split()  # list of words
        decoded_lst = decoded_words  # list of decoded words
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'decoded_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        if self._pointer_gen:
            to_write['p_gens'] = p_gens

        output_filename = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_filename, 'w') as f:
            json.dump(to_write, f)

        tf.logging.info('Wrote visualization data to %s', output_filename)


def print_results(article, abstract, decoded_output):
    """ Prints the article, the reference summary and the decoded summary to screen """
    print('---------------------------------------------------------------------------')
    tf.logging.info('ARTICLE:  %s', article)
    tf.logging.info('REFERENCE SUMMARY: %s', abstract)
    tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
    print('---------------------------------------------------------------------------')


def make_html_safe(s):
    """ Replace any angled brackets in string s to avoid interfering with HTML attention visualizer. """
    s.replace('<', '&lt;')
    s.replace('>', '&gt;')
    return s


def rouge_eval(ref_dir, dec_dir):
    """ Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict """
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """
    Log ROUGE results to screen and write to file.

    :param results_dict: dictionary returned by pyrouge
    :param dir_to_write: directory where we will write the results to
    :return:
    """
    log_str = ''
    for x in ['1', '2', 'l']:
        log_str += '\nROUGE-%s:\n' % x
        for y in ['f_score', 'recall', 'precision']:
            key = 'rouge_%s_%s' % (x, y)
            key_cb = key + '_cb'
            key_ce = key + '_ce'
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += '%s: %.4f with confidence interval (%.4f, %.4f)\n' % (key, val, val_cb, val_ce)

    tf.logging.info(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, 'ROUGE_results.txt')
    tf.logging.info('Writing final ROUGE results to %s...', results_file)
    with open(results_file, 'w') as f:
        f.write(log_str)


def get_decode_dir_name(ckpt_name, data_path, max_enc_steps, min_dec_steps, max_dec_steps, beam_size):
    """
    Make a descriptive name for the decode dir, including the name of the checkpoint
    we use to decode. This is called in single_pass mode.

    :param ckpt_name:
    :param data_path:
    :param max_enc_steps:
    :param min_dec_steps:
    :param max_dec_steps:
    :param beam_size:
    :return:
    """
    if 'train' in data_path:
        dataset = 'train'
    elif 'val' in data_path:
        dataset = 'val'
    elif 'test' in data_path:
        dataset = 'test'
    else:
        raise ValueError('FLAGS.data_path %s should contain one of train, val or test' % data_path)
    # noinspection SpellCheckingInspection
    dir_name = 'decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec' % \
               (dataset, max_enc_steps, beam_size, min_dec_steps, max_dec_steps)
    if ckpt_name is not None:
        dir_name += '_%s' % ckpt_name

    return dir_name
