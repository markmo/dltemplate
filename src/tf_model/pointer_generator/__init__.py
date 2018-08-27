from argparse import ArgumentParser
from collections import namedtuple
from common.util import load_hyperparams, merge_dict
from common.experimentalist import make_experiment_name, record_experiment, set_experiment_defaults
import os
import tensorflow as tf
from tf_model.pointer_generator.data import Vocab
from tf_model.pointer_generator.decode import BeamSearchDecoder
from tf_model.pointer_generator.model_setup import SummarizationModel
from tf_model.pointer_generator.util import Batcher, run_eval, setup_training


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    set_experiment_defaults(constants, {
        'experiment_name': make_experiment_name(),
        'model_name': 'summarization',
        'model_description': 'Pointer-generator network for text summarization',
        'model_type': 'RNN',
        'library': 'TensorFlow',
        'library_version': '1.2.1',
        'author_username': 'markmo',
        'author_uri': 'https://github.com/markmo'
    })
    record_experiment(constants)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting seq2seq_attention in %s mode...', constants['mode'])
    constants['log_root'] = os.path.join(constants['log_root'], constants['experiment_name'])
    if not os.path.exists(constants['log_root']):
        if constants['mode'] == 'train':
            os.makedirs(constants['log_root'])
        else:
            raise Exception("log_root %s doesn't exist. Run in train mode to create it." % constants['log_root'])

    vocab = Vocab(constants['vocab_path'], constants['vocab_size'])

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam,
    # so we need to make a batch of these hypotheses.
    if constants['mode'] == 'decode':
        constants['batch_size'] = constants['beam_size']

    # If single_pass=True, then check we're in decode mode
    if constants['single_pass'] and constants['mode'] != 'decode':
        raise Exception('The single_pass flag should only be True in decode mode.')

    # Make a namedtuple `config` with the hyperparameters the model needs
    hparams = ['mode', 'learning_rate', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
               'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps',
               'coverage', 'cov_loss_wt', 'pointer_gen', 'log_root']
    config_dict = {k: v for k, v in constants.items() if k in hparams}
    config = namedtuple('Hyperparams', config_dict.keys())(**config_dict)

    # Create a Batcher object that will create minibatches of data
    batcher = Batcher(constants['data_path'], vocab, config, single_pass=constants['single_pass'])

    tf.set_random_seed(111)
    # noinspection PyUnresolvedReferences
    if config.mode == 'train':
        print('Creating model...')
        model = SummarizationModel(config, vocab)
        setup_training(model, batcher, constants)
    elif config.mode == 'eval':
        model = SummarizationModel(config, vocab)
        run_eval(model, batcher, constants)
    elif config.mode == 'decode':
        # The model is configured with max_dec_steps=1 because we only ever
        # run one step of the decoder at a time (to do beam search). Note that
        # the batcher is initialized with max_dec_steps equal to, say 100,
        # because the batches need to contain the full summaries.
        # noinspection PyProtectedMember,PyUnresolvedReferences
        dec_model_conf = config._replace(max_dec_steps=1)
        model = SummarizationModel(dec_model_conf, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab, config)

        # decode indefinitely (unless single_pass=True, in which case decode the dataset exactly once.)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of ['train', 'eval', 'decode']")


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Pointer Generator Network')
    parser.add_argument('--mode', dest='mode', type=str, help='run mode (train, eval, decode)')
    parser.add_argument('--data-path', dest='data_path', type=str,
                        help='Path expression to tf.Example datafiles. '
                             'Can include wildcards to access multiple datafiles.')
    parser.add_argument('--vocab-path', dest='vocab_path', type=str, help='Path expression to text vocabulary file.')
    parser.add_argument('--log-root', dest='log_root', type=str, help='root directory for all logging')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name for experiment')
    parser.add_argument('--hidden-size', dest='hidden_dim', type=int, help='dimension of RNN hidden states')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--single-pass', dest='single_pass',
                        help='run eval on the full dataset using a fixed checkpoint', action='store_true')
    parser.set_defaults(single_pass=False)
    args = parser.parse_args()

    run(vars(args))
