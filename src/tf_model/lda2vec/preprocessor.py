from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import spacy
from tqdm import tqdm


class Preprocessor(object):

    def __init__(self, df, text_col, max_features=30000, max_len=10, window_size=5, nlp='en_core_web_sm',
                 bad=None, token_type='lower'):
        """

        :param df: (DataFrame) that has a text column
        :param text_col: (str) name of column in `df` that has text
        :param max_features: (int, optional) maximum number of unique words to keep
        :param max_len: (int, optional) maximum length of a document. Documents will
               be truncated at this length after tokenization but before computing
               skipgram pairs.
        :param window_size: (int, optional) size of sampling window (technically half-window).
        :param nlp: (str, optional) Spacy model to load (e.g. 'en', 'en_core_web_sm',
               'en_core_web_lg', or some custom model)
        :param bad: (set, optional) set of known bad characters to remove from the
               dataset.
        :param token_type: (str, optional) type of token to use, one of ['lower', 'lemma'].
               A value that isn't one of these options, will use the original text.
        """
        if bad is None:
            bad = {'ax>', '`@("', '---', '===', '^^^', 'AX>', 'GIZ', '--'}

        self.df = df
        self.text_col = text_col
        self.bad = bad
        self.token_type = token_type
        self.max_len = max_len
        self.max_features = max_features
        self.window_size = window_size

        # Disable parts of Spacy's pipeline, which really improves speed
        self.nlp = spacy.load(nlp, disable=['ner', 'tagger', 'parser'])

        self.n_docs = 0
        self.texts_clean = []
        self.tokenizer = None
        self.idx_data = None
        self.doc_lengths = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.freqs = []
        self.purged_docs = []
        self.skipgrams_df = None

    def clean(self, line):
        return ' '.join(w for w in line.split() if not any(t in w for t in self.bad))

    def tokenize_and_process(self):
        # Convert text column to list
        texts = self.df[self.text_col].values.tolist()

        # Clean texts
        texts = [str(self.clean(line)) for line in texts]

        self.n_docs = len(texts)
        self.texts_clean = []
        print('')
        print('------------ Tokenizing texts ------------')
        for i, doc in tqdm(enumerate(self.nlp.pipe(texts, n_threads=4))):
            # Variable for holding cleaned tokens (to be joined later)
            doc_texts = []
            for token in doc:
                if not token.like_email and not token.like_url:
                    if self.token_type == 'lemma':
                        doc_texts.append(token.lemma_)
                    elif self.token_type == 'lower':
                        doc_texts.append(token.lower_)
                    else:
                        doc_texts.append(token.text)

            self.texts_clean.append(' '.join(doc_texts))

        # Initialize a tokenizer and fit it with our cleaned texts
        self.tokenizer = Tokenizer(self.max_features, filters='', lower=False)
        self.tokenizer.fit_on_texts(self.texts_clean)

        # Get the idx data from the tokenizer
        self.idx_data = self.tokenizer.texts_to_sequences(self.texts_clean)

        # Limit length of data entries to `max_len`
        self.idx_data = [d[:self.max_len] for d in self.idx_data]

    def get_supplemental_data(self):
        # Get lengths of each doc
        self.doc_lengths = [len(d) for d in self.idx_data]

        # Get word-to-idx from Keras tokenizer
        # NOTE! Keras Tokenizer indexes from 1, 0 is reserved for PAD token
        # See https://github.com/keras-team/keras/issues/9637
        self.word_to_idx = self.tokenizer.word_index

        # Flip `word_to_idx` to get idx-to-word
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        # Vocab size should be at most `max_features` and will default to
        # `len(idx_to_word)` if less than that.
        self.vocab_size = min(self.max_features, len(self.idx_to_word))

        self.freqs = [0]
        for i in range(1, self.vocab_size + 1):
            token = self.idx_to_word[i]
            self.freqs.append(self.tokenizer.word_counts[token])

    def load_glove(self, embeddings_path):
        embed_indices = dict(get_coefs(*f.split(' ')) for f in open(embeddings_path, 'r'))
        return self._get_embeddings(embed_indices)

    def load_fasttext(self, embeddings_path):
        embed_indices = dict(get_coefs(*d.split(' ')) for d in open(embeddings_path, 'r') if len(d) > 100)
        return self._get_embeddings(embed_indices)

    def load_para(self, embeddings_path):
        embed_indices = dict(get_coefs(*d.split(' '))
                             for d in open(embeddings_path, mode='r', encoding='utf-8', errors='ignore')
                             if len(d) > 100)
        return self._get_embeddings(embed_indices)

    def _get_embeddings(self, embed_indices):
        all_embeds = np.stack(embed_indices.values())
        embed_mean, embed_std = all_embeds.mean(), all_embeds.std()
        embed_size = all_embeds.shape[1]

        embed_mat = np.random.normal(embed_mean, embed_std, (self.vocab_size + 1, embed_size))
        for word, idx in self.word_to_idx.items():
            if idx >= self.vocab_size:
                continue

            embed_vec = embed_indices.get(word)
            if embed_vec is not None:
                embed_mat[idx] = embed_vec

        return embed_mat

    def get_skipgrams(self):
        """
        Get all skipgram pairs as input to the LDA2Vec Model.

        Note: if a document has too few tokens to compute skipgrams, it
        is put into `purged_docs` and ignored.

        Values are stored in a DataFrame with the following columns:
        0: pivot idx
        1: context idx
        2: unique doc id - purged docs are not included. The unique doc id is
           used to create the embedding matrix.
        3: original doc id
        :return:
        """
        # list to hold skipgrams and associated metadata
        skipgram_data = []

        # list of indices (from original texts list in DataFrame) of purged docs
        self.purged_docs = []

        doc_id_counter = 0
        print('')
        print('------------ Getting skipgrams ------------')
        for i, t in tqdm(enumerate(self.idx_data)):
            pairs, _ = skipgrams(t, vocabulary_size=self.vocab_size, window_size=self.window_size,
                                 shuffle=True, negative_samples=0)
            if len(pairs) > 2:
                for pair in pairs:
                    temp = pair
                    # Append doc id
                    temp.append(doc_id_counter)
                    # Append doc index (from original texts list in DataFrame)
                    temp.append(i)
                    skipgram_data.append(temp)

                doc_id_counter += 1
            else:
                # Purge docs with less than 2 pairs
                self.purged_docs.append(i)

        self.skipgrams_df = pd.DataFrame(skipgram_data)

    def preprocess(self):
        self.tokenize_and_process()
        self.get_supplemental_data()
        self.get_skipgrams()

    def save_data(self, path, embed_mat=None):
        """
        Save all preprocessed data to given path. Optionally save the
        embedding matrix in the same path by passing in the `embed_mat`
        argument.

        The embedding matrix should have been created using one of:
        `load_glove`, `load_fasttext` or `load_para` methods. If not,
        the embedding matrix must use the same indices as in `word_to_idx`.

        :param path: (str) save path
        :param embed_mat:
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)

        path = Path(path)

        # if `embed_mat` is passed in, save it so long as `embed_mat.shape[0] == self.vocab_size`
        if isinstance(embed_mat, np.ndarray):
            assert embed_mat.shape[0] == self.vocab_size + 1, \
                ('embed_mat.shape[0] should equal '
                 '(vocab_size + 1): {0} != {1}').format(embed_mat.shape[0], self.vocab_size + 1)
            np.save(path / 'embedding_matrix', embed_mat)
        else:
            assert embed_mat is None, \
                'To save embeddings, it must be of type np.ndarray instead of {}'.format(type(embed_mat))

        # Save vocab dicts to file
        with open(path / 'idx_to_word.pkl', 'wb') as f:
            pickle.dump(self.idx_to_word, f)

        with open(path / 'word_to_idx.pkl', 'wb') as f:
            pickle.dump(self.word_to_idx, f)

        # np.save(path / 'doc_lengths', self.doc_lengths)
        np.save(path / 'doc_lengths', np.delete(self.doc_lengths, np.array(self.purged_docs)))
        np.save(path / 'freqs', self.freqs)
        self.skipgrams_df.to_csv(path / 'skipgrams.txt', sep='\t', index=False, header=None)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
