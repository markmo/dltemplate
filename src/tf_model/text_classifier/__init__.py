from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import os


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)



if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Text Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--embeddings-size', dest='emb_dim', type=int, help='embeddings size')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='dimension of RNN hidden states')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    args = parser.parse_args()

    run(vars(args))
