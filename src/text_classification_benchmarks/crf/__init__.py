from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import os
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples
from text_classification_benchmarks.data_loader import tokenize


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

    utterances_train, labels_train, utterances_val, labels_val, classes = load_datasets()
    tokens_train = tokenize(utterances_train)




if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run CRF Classifier')
    parser.add_argument('--samples', dest='n_samples', type=int, help='number samples')
    args = parser.parse_args()

    run(vars(args))
