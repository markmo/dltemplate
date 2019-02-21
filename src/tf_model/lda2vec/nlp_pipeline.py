import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import os
import pandas as pd
from sense2vec import Sense2VecComponent
import spacy
from spacy.attrs import *
import sys
import tensorflow as tf
import time


class NlpPipeline(object):

    # noinspection PyUnusedLocal
    def __init__(self, text_file, max_len, context=False, sep='\t', use_cols=None, tokenize_sents=False,
                 n_sents=4, google_news_word2vec_path=None, use_google_news=False, nlp=None,
                 nlp_object_path=None, vectors=None, skip_token='<SKIP>', merge=False, n_threads=2,
                 delete_punc=False, token_type='lower', skip_oov=False, save_tokenized_text_data=False,
                 bad_deps=('amod', 'compound'), texts=None, only_keep_alpha=False):
        """

        :param text_file: (str) Path to the line delimited text file
        :param max_len: (int) Length to limit/pad sequences to
        :param context: (bool, optional) If this parameter is False, we assume reviews delimited by new lines.
            If this parameter is True, we will load the data into a dataframe to extract labels/context.
        :param sep: (str, optional) Delimiter for csv/text files when context is True
        :param use_cols: (list[str], optional) List of column names to extract when context=True.
            *NOTE: The first name passed should be the column name for the text to be tokenized.*
        :param tokenize_sents: (bool, optional)
        :param n_sents: (int, optional)
        :param google_news_word2vec_path:
        :param use_google_news: (str) Path to Google News vectors bin file if `nlp` object has not been saved yet
        :param nlp: (optional) Pre-initialized Spacy NLP object
        :param nlp_object_path: (str) When passed, it will load the `nlp` object found in this path
        :param vectors: (optional)
        :param skip_token: (str, optional) Short documents will be padded with this token up until `max_len`
        :param merge: (bool, optional) When True, we will merge noun phrases and named entities into single tokens
        :param n_threads: (int, optional) Number of threads to parallelize the pipeline
        :param delete_punc: (bool, optional) When set to true, punctuation will be deleted when tokenizing
        :param token_type: (str, optional) String denoting type of token for tokenization.
            Options are "lower", "lemma", and "orth".
        :param skip_oov: (bool, optional) When set to true, it will replace out-of-vocabulary words with `skip_token`.
            Note: Setting this to false when planning to initialize random vectors will allow for learning
            the out-of-vocabulary words/phrases.
        :param save_tokenized_text_data: (bool, optional)
        :param bad_deps: (tuple[str])
        :param texts:
        :param only_keep_alpha:
        """
        self.google_news_word2vec_path = google_news_word2vec_path
        self.text_file = text_file
        self.max_len = max_len
        self.context = context
        self.sep = sep
        self.use_cols = use_cols
        self.skip_token = skip_token
        self.nlp = nlp
        self.merge = merge
        self.n_threads = n_threads
        self.delete_punc = delete_punc
        self.token_type = token_type
        self.skip_oov = skip_oov
        self.save_tokenized_text_data = save_tokenized_text_data
        self.bad_deps = bad_deps
        self.tokenize_sents = tokenize_sents
        self.n_sents = n_sents
        self.nlp_object_path = nlp_object_path
        self.vectors = vectors
        self.texts = texts
        self.tokenizing_new = False
        self.only_keep_alpha = only_keep_alpha
        self.model = None
        self.keys = []
        self.context_df = None
        self.n_docs = 0
        self.data = None
        self.purged_docs = []
        self.text_data = []
        self.uniques = None
        self.vocab = None
        self.hash_to_word = {}
        self.embed_matrix_tensor = None
        self.idx_data = None
        self.vocabulary = None
        self.idx_to_word = {}
        self.word_to_idx = {}
        self.context_desc = None
        self.seq_desc = None
        self.labels_desc = None
        self.doc_id = 0
        self.timer_dict = {}

        # If a spacy nlp object is not passed to init
        if self.nlp is None:
            if nlp_object_path:
                # Load nlp object from path provided
                self.nlp = spacy.load(self.nlp_object_path)
            elif self.google_news_word2vec_path:
                # Load Google News word2vec embeddings from binary file
                self.load_google_news_word2vec()
            elif self.vectors:
                # Use vectors path from saved dist-packages location
                self.nlp = spacy.load('en_core_web_lg', vectors=self.vectors)
            else:
                # If nothing is specified, load spacy model
                self.nlp = spacy.load('en_core_web_lg')

        self.tokenize()

    def load_google_news_word2vec(self):
        """
        To get frequencies:

        ::

            vocab_obj = model.vocab['word']
            vocab_obj.count

        :return:
        """
        # Load Google News word2vec embeddings using gensim
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.google_news_word2vec_path, binary=True)

        # Init blank english spacy nlp object
        self.nlp = spacy.load('en_core_web_lg', vectors=False)

        # Loop through range of all indexes and get words associated with each index.
        # The words in the keys list will correspond to the order of the Google embed matrix.
        self.keys = []
        for idx in range(3000000):
            word = self.model.index2word[idx]
            word = word.lower()
            self.keys.append(word)

            # Add the word to the nlp vocab
            self.nlp.vocab.strings.add(word)

        # Set the vectors for our nlp object to the Google news vectors
        # noinspection PyUnresolvedReferences
        self.nlp.vocab.vectors = spacy.vocab.Vectors(data=self.model.syn0, keys=self.keys)

    # noinspection PyDictCreation,PyUnboundLocalVariable
    def tokenize(self):
        if self.texts is None:
            if self.context is False:
                # Read in text data from text_file path
                self.texts = open(self.text_file).read().split('\n')
                self.texts = [str(t) for t in self.texts]
            else:
                filename, file_ext = os.path.splitext(self.text_file)
                if file_ext == '.json':
                    # Read in json data as dataframe
                    # noinspection PyUnresolvedReferences
                    df = pd.read_json(self.text_file, lines=True)
                else:
                    # Read in tabular data as dataframe
                    # noinspection PyUnresolvedReferences
                    df = pd.read_csv(self.text_file, sep=self.sep, usecols=self.use_cols)

                # Extract the text
                text_col_name = self.use_cols[0]
                self.texts = df[text_col_name].values.astype(str).tolist()

                # Small memory reduction by deleting this
                del df[text_col_name]
                self.context_df = df

        # Get number of documents supplied
        self.n_docs = len(self.texts)

        # Init data as a bunch of zeros - shape [n_docs, max_len]
        self.data = np.zeros((self.n_docs, self.max_len), dtype=np.uint64)

        if not self.tokenizing_new:
            # Add the skip token to the vocab, creating a unique hash for it
            self.nlp.vocab.strings.add(self.skip_token)
            self.skip_token = self.nlp.vocab.strings[self.skip_token]

        self.data[:] = self.skip_token

        # Make array to store row numbers of documents that must be deleted
        self.purged_docs = []

        # This array will hold tokenized text data if it is asked for
        if self.save_tokenized_text_data:
            self.text_data = []

        if self.tokenize_sents:
            self.sentence_tokenize()
            return

        # If we want to merge phrases, we add s2v component
        # to our pipe and it will do it for us.
        if self.merge:
            s2v = Sense2VecComponent('reddit_vectors-1.1.0')
            self.nlp.add_pip(s2v)

        for i, doc in enumerate(self.nlp.pipe(self.texts, n_threads=self.n_threads, batch_size=10000)):
            # noinspection PyBroadException
            try:
                # Create temp list for holding doc text
                if self.save_tokenized_text_data:
                    doc_text = []

                for token in doc:
                    # TODO - determine if you want to leave spaces or replace with underscores
                    # Replaces spaces between phrases with underscore
                    # text = token.text.replace(" ", "_")
                    # Get the string token for the given token type
                    if self.token_type == 'lower':
                        _token = token.lower_
                    elif self.token_type == 'lemma':
                        _token = token.lemma_
                    else:
                        _token = token.orth_

                    # Add token to spacy string list so we can use oov as known hash tokens
                    if token.is_oov:
                        self.nlp.vocab.strings.add(_token)

                    if self.save_tokenized_text_data:
                        doc_text.append(_token)

                if self.save_tokenized_text_data:
                    self.text_data.append(doc_text)

                # Options for how to tokenize
                if self.token_type == 'lower':
                    dat = doc.to_array([LOWER, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT, IS_ALPHA])
                elif self.token_type == 'lemma':
                    dat = doc.to_array([LEMMA, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT, IS_ALPHA])
                else:
                    dat = doc.to_array([ORTH, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT, IS_ALPHA])

                if len(dat) > 0:
                    assert dat.min() >= 0, 'Negative indices reserved for special tokens'
                    if self.skip_oov:
                        # Get indices of email, URL and oov tokens
                        idx = (dat[:, 1] > 0) | (dat[:, 2] > 0) | (dat[:, 3] > 0)
                    else:
                        # Get indices of email and URL tokens
                        idx = (dat[:, 1] > 0) | (dat[:, 2] > 0)

                    # Replace email and URL tokens with skip token
                    dat[idx] = self.skip_token

                    # Delete punctuation
                    if self.delete_punc:
                        delete = np.where(dat[:, 4] == 1)
                        dat = np.delete(dat, delete, 0)

                    if self.only_keep_alpha is True:
                        delete = np.where(dat[:, 5] == 0)
                        dat = np.delete(dat, delete, 0)

                    length = min(len(dat), self.max_len)
                    self.data[i, :length] = dat[:length, 0].ravel()

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('\n\n')
                print(exc_type, filename, exc_tb.tb_lineno)
                self.purged_docs.append(i)
                continue

        # If necessary, delete documents that failed to tokenize correctly.
        self.data = np.delete(self.data, self.purged_docs, 0).astype(np.uint64)

        # Unique tokens
        self.uniques = np.unique(self.data)

        # Saved Spacy Vocab
        self.vocab = self.nlp.vocab

        # Making an idx to word mapping for vocab
        self.hash_to_word = {}

        # Insert padding id into the hash
        self.hash_to_word[self.skip_token] = '<SKIP>'

        # If lemma, insert pronoun ID into the hash
        if self.token_type == 'lemma':
            self.hash_to_word[self.nlp.vocab.strings['-PRON-']] = '-PRON-'

        for v in self.uniques:
            if v != self.skip_token:
                # noinspection PyPep8,PyBroadException
                try:
                    if self.token_type == 'lower':
                        self.hash_to_word[v] = self.nlp.vocab[v].lower_
                    elif self.token_type == 'lemma':
                        self.hash_to_word[v] = self.nlp.vocab[v].lemma_
                    else:
                        self.hash_to_word[v] = self.nlp.vocab[v].orth_
                except Exception:
                    pass

    # noinspection PyDictCreation,PyUnboundLocalVariable
    def sentence_tokenize(self, ):
        # data has shape [n_docs, None, max_len]
        self.data = np.zeros([self.n_docs, self.n_sents, self.max_len], dtype=np.uint64)
        self.data[:] = self.skip_token

        i = 0
        try:
            for i, doc in enumerate(self.nlp.pipe(self.texts, n_threads=self.n_threads, batch_size=10000)):
                for j, sent in enumerate(doc.sents):
                    # We don't want to process more than num sentences for each doc.
                    # Limit the number of sentences tokenized to `n_sents`.
                    if j >= self.n_sents:
                        continue

                    # noinspection PyBroadException
                    try:
                        if self.merge:
                            # Make list to hold merged phrases. Necessary to avoid buggy spacy merge implementation.
                            phrase_list = []

                            # Merge noun phrases into single tokens
                            for phrase in list(sent.noun_chunks):
                                while len(phrase) > 1 and phrase[0].dep_ not in self.bad_deps:
                                    phrase = phrase[1:]

                                if len(phrase) > 1:
                                    phrase_list.append(phrase)

                            # Merge phrases into `sent` using `sent.merge`. `phrase.merge` breaks.
                            if len(phrase_list) > 0:
                                for _phrase in phrase_list:
                                    sent.merge(start_idx=_phrase[0].idx,
                                               end_idx=_phrase[len(_phrase) - 1].idx + len(_phrase[len(_phrase) - 1]),
                                               tag=_phrase[0].tag_,
                                               lemma='_'.join([token.text for token in _phrase]),
                                               ent_type=_phrase[0].ent_type_)

                            ent_list = []
                            for ent in sent.ents:
                                if len(ent) > 1:
                                    ent_list.append(ent)

                            # Merge entities into `sent` using `sent.merge`. `ent.merge` breaks.
                            if len(ent_list) > 0:
                                for _ent in ent_list:
                                    sent.merge(start_idx=_ent[0].idx,
                                               end_idx=_ent[len(_ent) - 1].idx + len(_ent[len(_ent) - 1]),
                                               tag=_ent.root.tag_,
                                               lemma='_'.join([token.text for token in _ent]),
                                               ent_type=_ent[0].ent_type_)

                        # Create temp list for holding doc text
                        if self.save_tokenized_text_data:
                            sent_text = []

                        for token in sent:
                            # Replaces spaces between phrases with underscore
                            # text = token.text.replace(' ', '_')

                            # Get the string token for the given token type
                            if self.token_type == 'lower':
                                _token = token.lower_
                            elif self.token_type == 'lemma':
                                _token = token.lemma_
                            else:
                                _token = token.orth_

                            # Add token to spacy string list so we can use oov as known hash tokens
                            if token.is_oov:
                                self.nlp.vocab.strings.add(_token)

                            if self.save_tokenized_text_data:
                                sent_text.append(_token)

                        if self.save_tokenized_text_data:
                            self.text_data.append(sent_text)

                        # Options for how to tokenize
                        if self.token_type == 'lower':
                            dat = sent.to_array([LOWER, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT])
                        elif self.token_type == 'lemma':
                            dat = sent.to_array([LEMMA, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT])
                        else:
                            dat = sent.to_array([ORTH, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT])

                        if len(dat) > 0:
                            assert dat.min() >= 0, 'Negative indices reserved for special tokens'
                            if self.skip_oov:
                                # Get indices of email, URL and oov tokens
                                idx = (dat[:, 1] > 0) | (dat[:, 2] > 0) | (dat[:, 3] > 0)
                            else:
                                # Get indices of email and URL tokens
                                idx = (dat[:, 1] > 0) | (dat[:, 2] > 0)

                            # Replace email and URL tokens with skip token
                            dat[idx] = self.skip_token

                            # Delete punctuation
                            if self.delete_punc:
                                delete = np.where(dat[:, 3] == 1)
                                dat = np.delete(dat, delete, 0)

                            length = min(len(dat), self.max_len)
                            self.data[i, j, :length] = dat[:length, 0].ravel()
                    except Exception:
                        print(sent)
                        print('Warning: document', i, 'broke! Likely due to spacy merge issues (#1547, #1474).')
                        self.purged_docs.append(i)
                        continue
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # filename = os.path.splitext(exc_tb.tb_frame.f_code.co_filename)[1]
            print(i, exc_type, e, exc_tb.tb_lineno)
            self.purged_docs.append(i)

        if len(self.purged_docs) > 0:
            # If necessary, delete documents that failed to tokenize correctly.
            self.data = np.delete(self.data, self.purged_docs, 0).astype(np.uint64)

        self.uniques = np.array([], dtype=np.uint64)

        # Get unique data by any means necessary for now...
        for doc_i in range(self.data.shape[0]):
            for sent_j in range(self.data[doc_i].shape[0]):
                self.uniques = np.append(self.uniques, self.data[doc_i][sent_j])

        # Saved spacy vocab
        self.vocab = self.nlp.vocab

        # Making an idx to word mapping for vocab
        self.hash_to_word = {}

        # Insert padding id into the hash
        self.hash_to_word[self.skip_token] = '<SKIP>'

        # If lemma, insert pronoun ID into the hash
        if self.token_type == 'lemma':
            self.hash_to_word[self.nlp.vocab.strings['-PRON-']] = '-PRON-'

        for v in self.uniques:
            if v != self.skip_token:
                # noinspection PyPep8,PyBroadException
                try:
                    if self.token_type == 'lower':
                        self.hash_to_word[v] = self.nlp.vocab[v].lower_
                    elif self.token_type == 'lemma':
                        self.hash_to_word[v] = self.nlp.vocab[v].lemma_
                    else:
                        self.hash_to_word[v] = self.nlp.vocab[v].orth_
                except Exception:
                    pass

    def compute_embed_matrix(self, random=False, embed_size=256, compute_tensor=False, tf_as_variable=True):
        """
        Computes the embedding matrix in a couple of ways. You can either initialize it randomly
        or you can load the embedding matrix from pretrained embeddings.

        Additionally, you can use this function to compute your embedding matrix as a tensorflow
        variable or tensorflow constant.

        The numpy embedding matrix will be stored in self.embed_matrix

        The tensorflow embed matrix will be stored in self.embed_matrix_tensor if compute_tf is True

        :param random: (bool, optional) If set to true, will initialize the embedding matrix randomly
        :param embed_size: (int, optional) If setting up random embedding matrix, you can control the embedding size.
        :param compute_tensor: (bool, optional) When set to True, it will turn the embedding matrix
            into a tf variable or constant. See `tf_as_variable` to control whether embed_matrix_tensor
            is a variable or constant.
        :param tf_as_variable: (bool, optional) If True AND compute_tf is True, this will save
            `embed_matrix_tensor` as a tf variable. If this is set to False, it will compute
            `embed_matrix_tensor` as a tf constant.
        :return:
        """
        self.unique, self.freqs = np.unique(self.data, return_counts=True)

        # Sort unique hash id values by frequency
        self.hash_ids = [x for _, x in sorted(zip(self.freqs, self.unique), reverse=True)]
        self.freqs = sorted(self.freqs, reverse=True)

        # Create word ids starting at 0
        self.word_ids = np.arange(len(self.hash_ids))

        self.hash_to_idx = dict(zip(self.hash_ids, self.word_ids))
        self.idx_to_hash = dict(zip(self.word_ids, self.hash_ids))

        # Generate random embedding instead of using pretrained embeddings
        if random:
            self.compute_idx_helpers()
            self.embed_size = embed_size
            self.vocab_size = len(self.unique)
            self.embed_matrix = np.random.uniform(-1, 1, [self.vocab_size, self.embed_size])
            if compute_tensor:
                self.compute_embed_tensor(variable=tf_as_variable)

            return

        self.vocab_size = len(self.uniques)

        # Initialize vector of zeros to compare to OOV vectors (which will all be zero)
        zeros = np.zeros(300)

        # Initialize vector to hold our embedding matrix
        embed_matrix = []

        self.vocabulary = np.array([])
        self.idx_to_word = {}

        # Loop through hash IDs. They are in order of highest frequency to lowest.
        for i, h in enumerate(self.hash_ids):
            # Extract word for given hash
            word = self.nlp.vocab.strings[h]

            # Append word to unique vocabulary list
            self.vocabulary = np.append(self.vocabulary, word)

            # Add key-value pair to `idx_to_word` dictionary
            self.idx_to_word[i] = word

            # Extract vector for the given hash id
            vector = self.nlp.vocab[h].vector

            # If the given vector is just zeros, it is out-of-vocabulary.
            if np.array_equal(zeros, vector):
                # TODO - get rid of this random uniform vector
                # If oov, init a random uniform vector
                vector = np.random.uniform(-1, 1, 300)

            # Append current vector to our embed matrix
            embed_matrix.append(vector)

        # Flip `idx_to_word` dictionary and save it to the class
        self.word_to_idx = {v: k for k, v in self.idx_to_word.items()}

        # Save np embed matrix to the class for later use
        self.embed_matrix = np.array(embed_matrix)

        # Get the embedding dimension size
        self.embed_size = self.embed_matrix.shape[1]

        if compute_tensor:
            self.compute_embed_tensor(variable=tf_as_variable)

    def compute_embed_tensor(self, variable=True):
        """

        :param variable: (bool, optional) If variable is set to True, it will compute a tensorflow variable.
            If False, it will compute a tensorflow constant.
        :return:
        """
        # Create tensor and variable for use in tensorflow
        embed_matrix_tensor = tf.convert_to_tensor(self.embed_matrix)
        if variable:
            # Create embed matrix as tf variable
            self.embed_matrix_tensor = tf.Variable(embed_matrix_tensor)
        else:
            # Create embed matrix as tf constant
            self.embed_matrix_tensor = tf.Constant(embed_matrix_tensor)

    def convert_data_to_word2vec_indices(self):
        # Uses `hash_to_idx` dictionary to map data to indices
        self.idx_data = np.vectorize(self.hash_to_idx.get)(self.data)

    def compute_idx_helpers(self):
        self.vocabulary = np.array([])
        self.idx_to_word = {}
        for i, h in enumerate(self.hash_ids):
            # Extract word for given hash
            word = self.nlp.vocab.strings[h]

            # Append word to unique vocabulary list
            self.vocabulary = np.append(self.vocabulary, word)

            # Add key-value pair to `idx_to_word` dictionary
            self.idx_to_word[i] = word

        self.word_to_idx = {v: k for k, v in self.idx_to_word.items()}

    def trim_zeros_from_idx_data(self, idx_data=None):
        """
        This will trim the tail zeros from the `idx_data` variable
        and replace the variable.

        :param idx_data:
        :return:
        """
        if idx_data is not None:
            self.idx_data = np.array([np.trim_zeros(a, trim='b') for a in idx_data])
        else:
            self.idx_data = np.array([np.trim_zeros(a, trim='b') for a in self.idx_data])

    def save_nlp_object(self, nlp_object_path):
        self.nlp.to_disk(nlp_object_path)

    def hash_seq_to_words(self, seq):
        """
        Pass this a single tokenized list of hash ids and it will
        translate it to words.

        :param seq:
        :return:
        """
        return ' '.join([self.hash_to_word[seq[i]] for i in range(seq.shape[0])])

    def load_gensim_word2vec(self, label=None, vector_size=128, window=5, min_count=5, workers=2):
        """
        Note: ensure you have `save_tokenized_text_data` set to True when running tokenizer

        :param label:
        :param vector_size:
        :param window:
        :param min_count:
        :param workers:
        :return:
        """
        doc2vec_data = []

        # If user supplies labels (in same length as number of docs), we can use those
        if label is None:
            label = np.arange(len(self.text_data)).tolist()

        # Loop through text data and format it for doc2vec compatibility
        for i, d in enumerate(self.text_data):
            doc2vec_data.append(TaggedDocument(d, [label[i]]))

        model = Doc2Vec(doc2vec_data, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        return model, doc2vec_data

    def make_example(self, seq, context=None):
        ex = tf.train.SequenceExample()

        # A non-sequential feature of our example: Replace with doc ids
        if context is None:
            context = self.doc_id

        # Note: This is an integer value. We will use these to supply doc context.
        ex.context.feature[self.context_desc].int64_list.value.append(context)

        # Feature lists for the two sequential features of our example
        feat_list_tokens = ex.feature_lists.feature_list[self.seq_desc]
        for token in seq:
            feat_list_tokens.feature.add().int64_list.value.append(token)

        return ex

    def make_example_with_labels(self, seq, labels, context=None):
        ex = tf.train.SequenceExample()

        # A non-sequential feature of our example: Replace with doc ids
        if context is None:
            context = self.doc_id

        # Note: This is an integer value. We will use these to supply doc context.
        ex.context.feature[self.context_desc].int64_list.value.append(context)

        # Feature lists for the two sequential features of our example
        feat_list_tokens = ex.feature_lists.feature_list[self.seq_desc]
        feat_list_labels = ex.feature_lists.feature_list[self.labels_desc]
        for token, label in zip(seq, labels):
            feat_list_tokens.feature.add().int64_list.value.append(token)
            feat_list_labels.feature.add().int64_list.value.append(label)

        return ex

    def write_data_to_tf_records(self, out_file, compression='GZIP',
                                 data=np.array([]), labels=np.array([]), context=np.array([]),
                                 seq_desc='tokens', labels_desc='labels', context_desc='doc_id'):
        if data.any() is False:
            data = self.idx_data

        # Name of the feature column for context
        self.context_desc = context_desc

        # Name of the sequence data feature column
        self.seq_desc = seq_desc

        # If labels provided, set the feature column name of these labels
        if labels.any():
            self.labels_desc = labels_desc

        # Create int to hold unique document id, as this will be our default context feature.
        self.doc_id = 1

        if compression == 'GZIP':
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        elif compression == 'ZLIB':
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        else:
            options = None

        # TODO - test
        with tf.python_io.TFRecordWriter(out_file, options=options) as writer:
            # Loop through data and create a serialized example of each
            for i, d in enumerate(data):
                # Get the serialized example
                if labels.any():
                    if context.any():
                        # If we have labels, we pass those to the function
                        ex = self.make_example_with_labels(d, labels[i], context=context[i])
                    else:
                        ex = self.make_example_with_labels(d, labels[i])
                else:
                    if context.any():
                        # If we have context, we pass this to the function
                        ex = self.make_example(d, context=context[i])
                    else:
                        ex = self.make_example(d)

                # Next, we write our serialized single example to file
                writer.write(ex.SerializeToString())

                self.doc_id += 1

    def tokenize_new_texts(self, texts, convert_to_idx=True):
        self.texts = texts
        self.tokenizing_new = True
        self.tokenize()

        # If `convert_to_idx` is True, return the list according to embedding indices instead of hashes
        if convert_to_idx:
            idx_data = np.array([])
            for doc in self.data:
                temp_doc = []
                for h in doc:
                    temp_doc.append(self.hash_to_idx.get(h, self.skip_token))

                if idx_data.any():
                    idx_data = np.vstack([idx_data, np.array([temp_doc])])
                else:
                    idx_data = np.array([temp_doc])

            return idx_data

    def timer(self, message, end=False):
        """
        Utility function to measure timing.

        :param message: (str) a message to annotate timings
        :param end: (bool, optional) If true, will check `timer_dict` at the message
            to see the time that was taken
        :return:
        """
        if not end:
            start = time.time()
            self.timer_dict[message] = start
        else:
            print('Took', time.time() - self.timer_dict[message], 'seconds to', message)
