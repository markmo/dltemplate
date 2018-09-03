from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import logging
import os
from tf_model.text_classifier.util import predict, train

logging.getLogger().setLevel(logging.INFO)


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    if constants['train']:
        logging.info('Training the model...')
        train(constants)
    else:
        logging.info('Making predictions...')
        predict(constants['trained_dir'])


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Text Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--embeddings-size', dest='emb_dim', type=int, help='embeddings size')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='dimension of RNN hidden states')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--trained-dir', dest='trained_dir', type=str, help='saved models dir')
    parser.add_argument('--train', dest='train', help='training mode', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
