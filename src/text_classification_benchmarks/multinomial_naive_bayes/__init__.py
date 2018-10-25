from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import numpy as np
import os
import pandas as pd
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples
from text_classification_benchmarks.metrics import perf_by_label, print_best_worst
from text_classification_benchmarks.multinomial_naive_bayes.model_setup import MultinomialNaiveBayesModel


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
    model = MultinomialNaiveBayesModel(x_train, y_train)

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

    # report_df = classification_report_to_df(y_val, y_pred, classes, counts_by_label, label_fixed_width=25)
    # sort_by_f1 = report_df.sort_values(by='f1_score', ascending=False)

    # print('Best 10:')
    # print(sort_by_f1[:10])
    # print()
    # print('Worst 10:')
    # print(sort_by_f1[sort_by_f1['f1_score'] != 0][-10:])
    # print()
    # print(report_df[-1:])

    stats = perf_by_label(y_val, y_pred, classes, counts_by_label)
    print('\nBest / Worst classes:')
    print_best_worst(stats, rounded=2, sort_column='f1_score', top_n=5, max_name_len=40)

    # labels = np.unique(y_val)
    # print(classification_report(y_val, y_pred, labels=labels, target_names=[classes[x] for x in labels]))
    # print(print_perf_by_label(stats, rounded=2, sort_column='f1_score'))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Multinomial Naive Bayes Classifier')
    parser.add_argument('--samples', dest='n_samples', type=int, help='number samples')
    args = parser.parse_args()

    run(vars(args))
