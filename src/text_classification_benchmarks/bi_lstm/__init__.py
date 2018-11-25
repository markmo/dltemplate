from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import csv
import os
from text_classification_benchmarks.bi_lstm.util import batch_iter, load_dataset, test, train


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    outdir = constants['outdir']
    run_dir = constants['run_dir']
    x_train, y_train, train_lengths, x_val, y_val, val_lengths, max_length, vocab_size, classes = \
        load_dataset(outdir, dirname=constants['data_dir'], vocab_name=constants['vocab_name'])

    if constants['test']:
        print('\nTesting...')
        preds = test(x_val, y_val, val_lengths, constants['test_batch_size'], run_dir, constants['checkpoint'],
                     constants['model_name'])

        # Save all predictions
        with open(os.path.join(run_dir, 'predictions.csv'), 'w', encoding='utf-8', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['True class', 'Prediction'])
            for i in range(len(preds)):
                csvwriter.writerow([y_val[i], preds[i]])

            print('Predictions saved to {}'.format(os.path.join(run_dir, 'predictions.csv')))

    else:
        print('\nTraining...')
        train_data = batch_iter(x_train, y_train, train_lengths, constants['batch_size'], constants['n_epochs'])
        train(train_data, x_val, y_val, val_lengths, len(classes), vocab_size,
              constants['n_hidden'], constants['n_layers'],
              constants['l2_reg_lambda'], constants['learning_rate'],
              constants['decay_steps'], constants['decay_rate'],
              constants['keep_prob'], outdir, constants['num_checkpoint'],
              constants['evaluate_every_steps'], constants['save_every_steps'],
              constants['summaries_name'], constants['model_name'])


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Bi-LSTM Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='number hidden layers')
    parser.add_argument('--embedding-size', dest='embedding_size', type=int, help='embedding size')
    parser.add_argument('--num-layers', dest='n_layers', type=int, help='number LSTM cells')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--outdir', dest='outdir', type=str, help='save directory')
    parser.add_argument('--rundir', dest='run_dir', type=str, help='run directory')
    parser.add_argument('--data-dir', dest='data_dir', type=str, help='relative path to data')
    parser.add_argument('--model-name', dest='model_name', type=str, help='model name')
    parser.add_argument('--vocab-name', dest='vocab_name', type=str, help='vocab name')
    parser.add_argument('--summaries-name', dest='summaries_name', type=str, help='summaries name')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str,
                        help='restore the graph from this model checkpoint')
    parser.add_argument('--test', dest='test',
                        help='run eval on the test dataset using a fixed checkpoint', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    run(vars(args))
