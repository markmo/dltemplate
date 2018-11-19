from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import os
from text_classification_benchmarks.char_cnn.model_setup import CharCNN
from text_classification_benchmarks.char_cnn.util import Data
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    train_df, val_df, test_df, classes = load_data(dirname=constants['data_dir'])
    train_df = remove_classes_with_too_few_examples(clean_data(train_df))
    val_df = remove_classes_with_too_few_examples(clean_data(val_df))
    n_classes = len(classes)
    alphabet = constants['alphabet']
    input_size = constants['input_size']
    training_data = Data(train_df.utterance.values, train_df.label.values, alphabet, input_size, n_classes)
    training_data.load_data()
    x_train, y_train = training_data.get_all_data()
    val_data = Data(val_df.utterance.values, val_df.label.values, alphabet, input_size, n_classes)
    val_data.load_data()
    x_val, y_val = val_data.get_all_data()
    model = CharCNN(input_size,
                    alphabet_size=constants['alphabet_size'],
                    embedding_size=constants['embedding_size'],
                    conv_layers=constants['conv_layers'],
                    fully_connected_layers=constants['fully_connected_layers'],
                    n_classes=n_classes,
                    keep_prob=constants['keep_prob'],
                    model_filename=constants['model_filename'],
                    optimizer=constants['optimizer'],
                    loss=constants['loss'])
    batch_size = constants['batch_size']
    if constants['test']:
        print('Testing...')
        loss, accuracy = model.test(x_val, y_val, batch_size)
        print('Loss: {0:.4f}, Accuracy: {1:.0%}'.format(loss, accuracy))
    else:
        print('Training...')
        model.train(x_train, y_train, x_val, y_val, n_epochs=constants['n_epochs'],
                    batch_size=batch_size)  # , checkpoint_every=constants['checkpoint_every'])


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Char-CNN Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--embedding-size', dest='embedding_size', type=int, help='embedding size')
    parser.add_argument('--model-filename', dest='model_filename', type=str, help='path to model file')
    parser.add_argument('--data-dir', dest='data_dir', type=str, help='relative path to data')
    parser.add_argument('--test', dest='test',
                        help='run eval on the test dataset using a fixed checkpoint', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    run(vars(args))
