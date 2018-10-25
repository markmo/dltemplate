from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples


def generate_tfidf_features(train_df, val_df, cutoff=5, ngram_range=2):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=cutoff, norm='l2',
                            encoding='latin-1', ngram_range=(1, ngram_range), stop_words='english')
    df: pd.DataFrame = pd.concat([train_df, val_df])
    df.reset_index(drop=True, inplace=True)
    features = tfidf.fit_transform(df.utterance).toarray()
    labels = df.label
    return features, labels, tfidf, df.index


def print_correlated_n_grams(features, labels, tfidf, classes, label_idx):
    features_chi2 = chi2(features, labels == label_idx)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print('Label:', classes[label_idx])
    print('\tMost correlated unigrams:', ', '.join(unigrams[-3:]))
    print('\tMost correlated bigrams:', ', '.join(bigrams[-3:]))
    print()


def show_relevant_terms(features, labels, tfidf, classes, every=10):
    for i in range(len(classes)):
        if i % every == 0:  # show for every nth label
            print_correlated_n_grams(features, labels, tfidf, classes, i)


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    train_df, val_df, test_df, classes = load_data()
    train_df = remove_classes_with_too_few_examples(clean_data(train_df))
    val_df = remove_classes_with_too_few_examples(clean_data(val_df))
    features, labels, tfidf, _ = generate_tfidf_features(train_df, val_df,
                                                         cutoff=constants['cutoff'],
                                                         ngram_range=constants['ngram_range'])
    print('Number Utterances: {}, Features: {}'.format(*features.shape))
    show_relevant_terms(features, labels, tfidf, classes, every=20)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run TF-IDF Bag of Words (BOW) Classifier')
    parser.add_argument('--cutoff', dest='cutoff', type=int, help='document frequency threshold')
    parser.add_argument('--ngrams', dest='ngram_range', type=int, help='ngram range')
    args = parser.parse_args()

    run(vars(args))
