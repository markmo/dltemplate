""" This file contains code to process data into batches """
import numpy as np
import os
from queue import Queue
from random import shuffle
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tf_model.pointer_generator.data import abstract2ids, abstract2sents, article2ids, example_generator
from tf_model.pointer_generator.data import PAD_TOKEN, START_DECODING, STOP_DECODING
from threading import Thread
import time


class Batch(object):
    """ Class representing a mini-batch of train/val/test examples for text summarization. """

    def __init__(self, example_list, config, vocab):
        """
        Turns the example_list into a Batch object.

        :param example_list: list of Example objects
        :param config: hyperparameters
        :param vocab: Vocabulary object
        """
        self.enc_batch = None
        self.enc_lens = None
        self.enc_padding_mask = None
        self.max_art_oovs = 0
        self.art_oovs = []
        self.enc_batch_extend_vocab = None
        self.dec_batch = None
        self.target_batch = None
        self.dec_padding_mask = None
        self.original_articles = []
        self.original_abstracts = []
        self.original_abstracts_sents = []
        self.pad_id = vocab.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list, config)  # initialize the input to the encoder
        self.init_decoder_seq(example_list, config)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list, config):
        """
        Initializes the following:
            self.enc_batch: numpy array of shape (batch_size, <=max_enc_steps) containing integer ids
                            (all OOVs represented by UNK id), padded to length of longest sequence in
                            the batch
            self.enc_lens: numpy array of shape (batch_size) containing integers. The (truncated) length
                           of each encoder input sequence (pre-padding).
            self.enc_padding_mask: numpy array of shape (batch_size, <=max_enc_steps), containing 1s and
                                   0s. 1s correspond to real tokens in enc_batch and target_batch; 0s
                                   correspond to padding.

        If config.pointer_gen, additionally initializes the following:
            self.max_art_oovs: maximum number of in-article OOVs in the batch
            self.art_oovs: list of lists of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab: Same as self.enc_batch, but in-article OOVs are represented by
                                         their temporary article OOV number.

        :param example_list:
        :param config:
        :return:
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch
        # because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros([config.batch_size, max_enc_seq_len], dtype=np.int32)
        self.enc_lens = np.zeros([config.batch_size], dtype=np.int32)
        self.enc_padding_mask = np.zeros((config.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])

            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]

            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((config.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list, config):
        """
        Initializes the following:
            self.dec_batch: numpy array of shape (batch_size, max_dec_steps), containing integer ids
                            as input for the decoder, padded to max_dec_steps length.
            self.target_batch: numpy array of shape (batch_size, max_dec_steps), containing integer ids
                               for the target sequence, padded to max_dec_steps length.
            self.dec_padding_mask: numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s.
                                   1s correspond to real tokens in dec_batch and target_batch; 0s correspond
                                   to padding.
        :param example_list:
        :param config:
        :return:
        """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension
        # = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is
        # possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to
        # upgrade to that.
        self.dec_batch = np.zeros((config.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((config.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((config.batch_size, config.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        """ Store the original article and abstract strings in the Batch object """
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists


class Batcher(object):
    """ A class to generate mini-batches of data. Buckets examples together based on length of the encoder sequence. """

    MAX_BATCH_QUEUE = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, config, single_pass):
        """
        Initialize the batcher. Start threads that process the data into batches.

        :param data_path: tf.Example file pattern
        :param vocab: Vocabulary object
        :param config: hyperparameters
        :param single_pass: If True, run through the dataset exactly once (useful for
               when you want to run evaluation on the dev or test set). Otherwise generate
               random batches indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._config = config
        self._single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of Examples
        # waiting to be batched
        self._batch_queue = Queue(self.MAX_BATCH_QUEUE)
        self._example_queue = Queue(self.MAX_BATCH_QUEUE * config.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            # just one thread, so we read through the dataset just once
            self._num_example_q_threads = 1

            # just one thread to batch examples
            self._num_batch_q_threads = 1

            # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._bucketing_cache_size = 1

            # this will tell us when we're finished reading the dataset
            self._finished_reading = False
        else:
            # num threads to fill example queue
            self._num_example_q_threads = 16

            # num threads to fill batch queue
            self._num_batch_q_threads = 4

            # num batches-worth of examples to load into cache before bucketing
            self._bucketing_cache_size = 100

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()

        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they die
        # We don't want a watcher in single_pass mode because the threads shouldn't run forever
        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        """
        Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example repeated many times (beam_size);
        this is necessary for beam search.

        :return: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch. '
                               'Bucket queue size: %i, Input queue size: %i',
                               self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info('Finished reading dataset in single_pass mode.')
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        """ Reads data from file and processes into Examples which are then placed into the example queue. """
        input_gen = self.text_generator(example_generator(self._data_path, self._single_pass))

        while True:
            try:
                # read the next example from file. article and abstract are both strings.
                article, abstract = next(input_gen)
            except StopIteration:  # if there are no more examples:
                tf.logging.info('The example generator for this example-queue-filling-thread has exhausted data.')
                if self._single_pass:
                    tf.logging.info("single_pass mode is on, so we've finished reading dataset. "
                                    "This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception('single_pass mode is off but the example generator is out of data; error.')

            # Use the <s> and </s> tags in abstract to get a list of sentences.
            abstract_sentences = [sent.strip() for sent in abstract2sents(abstract)]
            example = Example(article, abstract_sentences, self._vocab, self._config)  # Process into an Example.
            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        """
        Takes Examples out of example queue, sorts them by encoder sequence length, processes
        into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        :return:
        """
        while True:
            if self._config.mode != 'decode':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self._config.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())

                # sort by length of encoder sequence
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)

                # Group the sorted Examples into batches, optionally shuffle the batches,
                # and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._config.batch_size):
                    batches.append(inputs[i:i + self._config.batch_size])

                if not self._single_pass:
                    shuffle(batches)

                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._config, self._vocab))

            else:  # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in range(self._config.batch_size)]
                self._batch_queue.put(Batch(b, self._config, self._vocab))

    def watch_threads(self):
        """ Watch example queue and batch queue threads and restart if dead. """
        while True:
            time.sleep(60)
            for i, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[i] = new_t
                    new_t.daemon = True
                    new_t.start()

            for i, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[i] = new_t
                    new_t.daemon = True
                    new_t.start()

    # noinspection PyMethodMayBeStatic
    def text_generator(self, example_gen):
        """
        Generates article and abstract text from tf.Example.

        :param example_gen: a generator of tf.Examples from file. See data.example_generator
        :return:
        """
        while True:
            eg = next(example_gen)  # e is a tf.Example
            try:
                # the article text was saved under the key 'article' in the data files
                article_text = eg.features.feature['article'].bytes_list.value[0].decode()

                # the abstract text was saved under the key 'abstract' in the data files
                abstract_text = eg.features.feature['abstract'].bytes_list.value[0].decode()
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue

            if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield article_text, abstract_text


# noinspection SpellCheckingInspection
class Example(object):
    """ Class representing a train/val/test example for text summarization. """

    def __init__(self, article, abstract_sentences, vocab, config):
        """
        Initializes the Example, performing tokenization and truncation to produce
        the encoder, decoder and target sequences, which are stored in self.

        :param article: (str) source text, each token is separated by a single space
        :param abstract_sentences: list of strings, one per abstract sentence. In each
               sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param config: hyperparameters
        """
        self.config = config

        # Get ids of special tokens
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]

        self.enc_len = len(article_words)  # store the length after truncation but before padding

        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words]

        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings

        # list of word ids; OOVs are represented by the id for UNK token
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps,
                                                                 start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their
            # temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)

            # Get a version of the reference summary where in-article OOVs are represented by
            # their temporary article OOV id
            abs_ids_extend_vocab = abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps,
                                                        start_decoding, stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    # noinspection PyMethodMayBeStatic
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """
        Given the reference summary as a sequence of tokens, return the input sequence
        for the decoder, and the target sequence which we will use to calculate loss.
        The sequence will be truncated if it is longer than max_len. The input sequence
        must start with the start_id and the target sequence must end with the stop_id
        (but not if it's been truncated).

        :param sequence: list of ids (integers)
        :param max_len: (int)
        :param start_id: (int)
        :param stop_id: (int)
        :return:
            inp: sequence length <= max_len starting with start_id
            target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end token
        else:  # no truncation
            target.append(stop_id)  # end token

        assert len(inp) == len(target)

        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """ Pad decoder input and target sequences with pad_id up to max_len. """
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)

        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """ Pad the encoder input sequence with pad_id up to max_len. """
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)

        if self.config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """
    Calculate the running average loss using exponential decay.

    Used to implement early stopping w.r.t. a smoother loss curve than the raw loss curve.

    :param loss: loss on the most recent eval step
    :param running_avg_loss: running_avg_loss so far
    :param summary_writer: FileWriter object to write for TensorBoard
    :param step: training step
    :param decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.
    :return:
        running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:
        # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss *= decay + (1 - decay) * loss

    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % decay
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def convert_to_coverage_model(log_root):
    """ Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint """
    tf.logging.info('Converting non-coverage model to coverage model...')

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=get_config())
    print('initializing everything...')
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if 'coverage' not in v.name and 'Adagrad' not in v.name])
    print('restoring non-coverage variables...')
    curr_ckpt = load_ckpt(saver, sess, log_root)
    print('Restored!')

    # save this model and quit
    new_filename = curr_ckpt + '_cov_init'
    print('Saving model to %s...' % new_filename)
    new_saver = tf.train.Saver()  # this one will save all variables that now exist
    new_saver.save(sess, new_filename)
    print('Saved!')
    exit()


def get_config():
    """ Returns config for tf.session """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


# noinspection PyBroadException
def load_ckpt(saver, sess, log_root, ckpt_dir='train'):
    """
    Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it
    to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name.

    :param saver:
    :param sess:
    :param log_root:
    :param ckpt_dir:
    :return:
    """
    while True:
        try:
            latest_filename = 'checkpoint_best' if ckpt_dir == 'eval' else None
            ckpt_dir = os.path.join(log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception:
            tf.logging.info('Failed to load checkpoint from %s. Sleeping for %i secs...', ckpt_dir, 10)
            time.sleep(10)


def restore_best_model(log_root):
    """ Load best model file from eval directory, add variables for adagrad, and save to train directory """
    tf.logging.info('Restoring best model for training...')

    # Initialize all vars in the model
    sess = tf.Session(config=get_config())
    print('Initializing all variables...')
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if 'Adagrad' not in v.name])
    print('Restoring all non-adagrad variables from best model in eval dir...')
    curr_ckpt = load_ckpt(saver, sess, log_root, 'eval')
    print('Restored %s.' % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split('/')[-1].replace('bestmodel', 'model')
    new_filename = os.path.join(log_root, 'train', new_model_name)
    print('Saving model to %s...' % new_filename)
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including the Adagrad variables
    new_saver.save(sess, new_filename)
    print('Saved!')
    exit()


def run_eval(model, batcher, constants):
    """
    Repeatedly runs eval iterations, logging to screen and writing summaries.
    Saves the model with the best loss seen so far.

    :param model:
    :param batcher:
    :param constants:
    :return:
    """
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)  # keep 3 best checkpoints at a time
    sess = tf.Session(config=get_config())
    eval_dir = os.path.join(constants['log_root'], 'eval')  # make a subdir of the root dir for eval data
    best_model_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)

    # the eval job keeps a smoother running average loss to tell it when to implement early stopping
    running_avg_loss = 0
    best_loss = None  # hold the best loss achieved so far
    while True:
        load_ckpt(saver, sess, constants['log_root'])  # load a new checkpoint
        batch = batcher.next_batch()

        # run eval on the batch
        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        tf.logging.info('seconds for batch: %.2f', t1 - t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if constants['coverage']:
            coverage_loss = results['coverage_loss']
            tf.logging.info('coverage_loss: %f', coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir.
        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s',
                            running_avg_loss, best_model_save_path)
            saver.save(sess, best_model_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()


def run_training(model, batcher, sess_context_manager, summary_writer, constants):
    """ Repeatedly runs training iterations, logging loss to screen and writing summaries """
    tf.logging.info('Starting run_training...')
    with sess_context_manager as sess:
        if constants['debug']:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        while True:  # repeats until interrupted
            batch = batcher.next_batch()
            tf.logging.info('running training step...')
            t0 = time.time()
            results = model.run_train_step(sess, batch)
            t1 = time.time()
            tf.logging.info('seconds for training step: %.3f', t1 - t0)
            loss = results['loss']
            tf.logging.info('loss: %f', loss)  # print the loss to screen
            if not np.isfinite(loss):
                raise Exception('Loss is not finite. Stopping.')

            if constants['coverage']:
                coverage_loss = results['coverage_loss']
                tf.logging.info('coverage_loss: %f', coverage_loss)  # print the coverage loss to screen

            # get the summaries and iteration number so we can write summaries to TensorBoard
            summaries = results['summaries']  # write these summaries to TensorBoard using summary_writer
            train_step = results['global_step']  # we need this to update our running average loss
            summary_writer.add_summary(summaries, train_step)  # write the summaries
            if train_step % 100 == 0:  # flush the summary writer every so often
                summary_writer.flush()


def setup_training(model, batcher, constants):
    """ Setup before starting training """
    train_dir = os.path.join(constants['log_root'], 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    model.build_graph()
    if constants['convert_to_coverage_model']:
        assert constants['coverage'], 'To convert your non-coverage model to a coverage model, ' \
                                      'run with convert_to_coverage_model=True and coverage=True'
        convert_to_coverage_model(constants['log_root'])

    if constants['restore_best_model']:
        restore_best_model(constants['log_root'])

    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    supervisor = tf.train.Supervisor(logdir=train_dir,
                                     is_chief=True,
                                     saver=saver,
                                     summary_op=None,
                                     save_summaries_secs=60,  # save summaries for TensorBoard every 60 secs
                                     save_model_secs=60,      # checkpoint every 60 secs
                                     global_step=model.global_step)
    summary_writer = supervisor.summary_writer
    tf.logging.info('Preparing or waiting for session...')
    sess_context_manager = supervisor.prepare_or_wait_for_session(config=get_config())
    tf.logging.info('Created session!')
    try:
        # this is an infinite loop until interrupted
        run_training(model, batcher, sess_context_manager, summary_writer, constants)
    except KeyboardInterrupt:
        tf.logging.info('Caught keyboard interrupt on worker. Stopping supervisor...')
        supervisor.stop()
