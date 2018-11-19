from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import numpy as np
import os
import pandas as pd
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples
from text_classification_benchmarks.metrics import perf_by_label, print_best_worst
from text_classification_benchmarks.logreg.model_setup import LogisticRegressionModel


def load_datasets():
    train_df, val_df, test_df, classes = load_data()
    train_df = remove_classes_with_too_few_examples(clean_data(train_df))
    val_df = remove_classes_with_too_few_examples(clean_data(val_df))
    x_train, y_train = train_df.utterance, train_df.label
    x_val, y_val = val_df.utterance, val_df.label
    return x_train, y_train, x_val, y_val, classes


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    x_train, y_train, x_val, y_val, classes = load_datasets()
    model = LogisticRegressionModel(x_train, y_train)

    def print_prediction(sample_idx):
        utterance = x_val[sample_idx]
        print('Utterance:', utterance)
        print('Actual:', classes[y_val[sample_idx]])
        print('Predicted:', classes[model.predict(utterance)])

    print('\nSample predictions:')
    samples = np.random.choice(x_val.index, size=constants['n_samples'])
    for i in samples:
        print_prediction(i)
        print()

    y_pred = [model.predict(utterance) for utterance in x_val]

    train_df = pd.DataFrame({'utterance': x_train, 'label': y_train})
    counts_by_label = train_df.groupby('label').utterance.count()

    stats = perf_by_label(y_val, y_pred, classes, counts_by_label)
    print('\nBest / Worst classes:')
    print_best_worst(stats, rounded=2, sort_column='f1_score', top_n=5, max_name_len=40)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Logistic Regression Classifier')
    parser.add_argument('--samples', dest='n_samples', type=int, help='number samples')
    args = parser.parse_args()

    run(vars(args))
