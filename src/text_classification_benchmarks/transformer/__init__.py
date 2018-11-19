from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import numpy as np
import os
from text_classification_benchmarks.metrics import perf_summary, print_perf_summary
from text_classification_benchmarks.transformer.util import create_vocab_labels_file, predict, train


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    classes_txt = '/Users/d777710/src/DeepLearning/dltemplate/src/text_classification_benchmarks/fastai/classes.txt'
    classes = np.genfromtxt(classes_txt, dtype=str)
    n_classes = len(classes)
    vocab_labels_filename = constants['vocab_labels_filename']
    if not os.path.exists(vocab_labels_filename):
        create_vocab_labels_file(vocab_labels_filename, classes)

    if constants['train']:
        print('Training...')
        train(constants['training_data_path'],
              n_classes,
              constants['learning_rate'],
              constants['batch_size'],
              constants['n_epochs'],
              constants['decay_steps'],
              constants['decay_rate'],
              constants['max_seq_len'],
              constants['embed_size'],
              constants['d_model'],
              constants['d_k'],
              constants['d_v'],
              constants['h'],
              constants['n_layers'],
              constants['l2_lambda'],
              constants['keep_prob'],
              constants['checkpoint_dir'],
              constants['use_embedding'],
              vocab_labels_filename,
              constants['word2vec_filename'],
              constants['validate_step'],
              constants['is_multilabel'])
    else:
        print('Testing...')
        result, labels = predict(constants['test_file'],
                                 n_classes,
                                 constants['learning_rate'],
                                 constants['batch_size'],
                                 constants['decay_steps'],
                                 constants['decay_rate'],
                                 constants['max_seq_len'],
                                 constants['embed_size'],
                                 constants['d_model'],
                                 constants['d_k'],
                                 constants['d_v'],
                                 constants['h'],
                                 constants['n_layers'],
                                 constants['l2_lambda'],
                                 constants['checkpoint_dir'],
                                 vocab_labels_filename,
                                 constants['word2vec_filename'])
        intents = [x[1][0] for x in result]
        preds = [list(classes).index(x) for x in intents]
        print('labels length:', len(labels), 'preds length:', len(preds))
        y_true = [list(classes).index(x) for x in labels]
        print(y_true)
        print(preds)
        stats = perf_summary(y_true, preds)
        print_perf_summary(stats, rounded=2)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Transformer Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--embedding-size', dest='embed_size', type=int, help='embedding size')
    parser.add_argument('--max-seq-len', dest='max_seq_len', type=int, help='maximum sequence length')
    parser.add_argument('--num-layers', dest='n_layers', type=int, help='number transformer layers')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--keep-prob', dest='keep_prob', type=float, help='dropout keep probability')
    parser.add_argument('--training-data-path', dest='training_data_path', type=str,
                        help='path to training data')
    parser.add_argument('--test-data-path', dest='test_file', type=str,
                        help='path to testing data')
    parser.add_argument('--vocab-labels-path', dest='vocab_labels_filename', type=str,
                        help='path to vocab labels file')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str,
                        help='checkpoint directory from training run')
    parser.add_argument('--word2vec-filename', dest='word2vec_filename', type=str,
                        help='path to word2vec embeddings file')
    parser.add_argument('--train', dest='train', help='run training', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()
    args_dict = vars(args)

    run(args_dict)
