from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]|@,;]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    text = ' '.join(filter(lambda x: x not in STOPWORDS, re.split(r'\s+', text)))
    return text


def get_counts(rows):
    """

    :param rows: list of lists or list of text
    :return: dict of items and counts
    """
    if isinstance(rows, (list, np.ndarray)):
        if isinstance(rows[0], (list, np.ndarray, tuple)):
            return dict(Counter(x for row in rows for x in row))
        elif isinstance(rows[0], str):
            return dict(Counter(x for row in rows for x in row.split()))

    raise ValueError('argument must be a list of lists or a list of text')


def top_n(freq_dict, n):
    """

    :param freq_dict: dict of items and counts
    :param n: number of items to return
    :return: list of top n items
    """
    return [x for x, _ in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:n]]


def map_words_to_index(words):
    return {w: i for i, w in enumerate(words)}


def map_index_to_words(words):
    return {i: w for i, w in enumerate(words)}


def to_bag_of_words_vector(text, words_to_index, dict_size):
    """

    :param text: a string
    :param words_to_index:
    :param dict_size: vocab size
    :return: a bag-of-words vector representation of `text`
    """
    v = np.zeros(dict_size)
    for w in text.split():
        if w in words_to_index:
            v[words_to_index[w]] = 1

    return v


def to_sparse_matrix(rows, words_to_index, dict_size):
    return sp_sparse.vstack(
        [sp_sparse.csr_matrix(to_bag_of_words_vector(text, words_to_index, dict_size)) for text in rows]
    )


def get_tf_idf_matrix(texts):
    # `min_df`, minimum document frequency - cut-off threshold for low frequency n-grams
    # `max_df`, maximum document frequency - cut-off threshold for high frequency n-grams such as stop words
    # `ngram_range` - which n-grams should be used in this bag-of-words (BOW) representation
    tf_idf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    features = tf_idf.fit_transform(texts)
    return pd.DataFrame(features.todense(), columns=tf_idf.get_feature_names())


def tfidf_features(x_train, x_val, x_test, min_df=5, max_df=0.9, ngram_range=None):
    """
    Create TF-IDF vectorizer with a proper parameters choice
    Fit the vectorizer on the train set
    Transform the train, test, and val sets and return the result

    :param x_train:
    :param x_val:
    :param x_test:
    :param min_df:
    :param max_df:
    :param ngram_range:
    :return: TF-IDF vectorized representation of each sample and vocabulary
    """
    if ngram_range is None:
        ngram_range = (1, 2)

    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    x_train = vectorizer.fit_transform(x_train)
    x_val = vectorizer.transform(x_val)
    x_test = vectorizer.transform(x_test)

    return x_train, x_val, x_test, vectorizer.vocabulary_
