from argparse import ArgumentParser
from collections import namedtuple
from common.util import load_hyperparams, merge_dict
import importlib
import os
import tensorflow as tf
from tf_model.neural_turing_machine.model_setup import create_ntm


# noinspection PyUnresolvedReferences
def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    config = namedtuple('Config', constants.keys())(*constants.values())

    with tf.device('/cpu:0'), tf.Session() as sess:
        try:
            task = importlib.import_module('tasks.%s' % config.task)
        except ImportError:
            print("task '%s' does not have implementation" % config.task)
            raise

        if config.is_train:
            cell, ntm = create_ntm(config, sess)
            task.train(ntm, config, sess)
        else:
            cell, ntm = create_ntm(config, sess, forward_only=True)

        ntm.load(config.checkpoint_dir, config.task)

        if config.task == 'copy':
            task.run(ntm, int(config.test_max_length * 1/3), sess)
            print()
            task.run(ntm, int(config.test_max_length * 2/3), sess)
            print()
            task.run(ntm, int(config.test_max_length * 3/3), sess)
        else:
            task.run(ntm, int(config.test_max_length), sess)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run NTM')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--hidden-size', dest='controller_dim', type=int, help='dimension of LSTM controller')
    parser.add_argument('--min-length', dest='min_length', type=int, help='minimum length of input sequence')
    parser.add_argument('--max-length', dest='max_length', type=int, help='maximum length of input sequence')
    parser.add_argument('--retrain', dest='is_train', help='train model', action='store_true')
    parser.set_defaults(is_train=False)
    args = parser.parse_args()

    run(vars(args))
