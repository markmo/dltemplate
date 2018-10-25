from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples
from text_classification_benchmarks.metrics import perf_summary, print_perf_summary
from text_classification_benchmarks.metrics import perf_by_label, print_perf_by_label
from text_classification_benchmarks.nbsvm.model_setup import NBSVM


def train_and_test(train_df, val_df, ngram_range=(1, 3)):
    print('Vectorizing...')
    vect = CountVectorizer()
    model = NBSVM()
    print('ngram_range:', ngram_range)

    # Create pipeline
    pipeline = Pipeline([('vect', vect), ('nbsvm', model)])
    params = {
        'vect__token_pattern': r'\S+',
        'vect__ngram_range': ngram_range,
        'vect__binary': True
    }
    pipeline.set_params(**params)

    print('Fitting...')
    pipeline.fit(train_df.utterance, train_df.label)

    print('Classifying...')
    preds = pipeline.predict(val_df.utterance)
    return preds, model


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    train_df, val_df, test_df, classes = load_data()
    train_df = remove_classes_with_too_few_examples(clean_data(train_df))
    val_df = remove_classes_with_too_few_examples(clean_data(val_df))

    preds, _ = train_and_test(train_df, val_df, ngram_range=constants['ngram_range'])

    stats = perf_summary(val_df.label, preds)
    print_perf_summary(stats, rounded=2)

    counts_by_label = train_df.groupby('label').utterance.count()
    print(print_perf_by_label(perf_by_label(val_df.label, preds, classes, counts_by_label), rounded=2))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run NBSVM Classifier')
    parser.add_argument('--ngrams', dest='ngram_range', help='ngram range')
    args = parser.parse_args()
    if args.ngram_range:
        # noinspection PyUnresolvedReferences
        args.ngram_range = tuple(int(x) for x in args.ngram_range.split(','))

    run(vars(args))
