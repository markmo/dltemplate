from argparse import ArgumentParser
from common.model_util import load_hyperparams, merge_dict
import gensim
import numpy as np
import os
import pandas as pd
import spacy
from tensorflow.contrib import learn
from tf_model.question_detector.util import load_data, preprocess, save_eval_to_csv, test, train

DATA_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/data/')


def convert_text_to_pos(text, nlp):
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return ' '.join(pos)


def train_word2vec_model(x):
    model = gensim.models.Word2Vec(x, min_count=1, size=300)
    return model


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    classes = np.array([0, 1])
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'train_pos.csv')):
        print('Loading POS data')
        train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train_pos.csv'), header=0, index_col=0)
        val_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'val_pos.csv'), header=0, index_col=0)
        test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_pos.csv'), header=0, index_col=0)
    else:
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'train.csv')):
            print('Loading text data')
            train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train.csv'), header=0, index_col=0)
            val_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'val.csv'), header=0, index_col=0)
            test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test.csv'), header=0, index_col=0)
        else:
            print('Creating text data from Yahoo dataset')
            train_df, val_df, test_df, _ = load_data(DATA_DIR + 'yahoo_non_factoid_qa/nfL6.json')
            train_df.to_csv(os.path.join(os.path.dirname(__file__), 'train.csv'))
            val_df.to_csv(os.path.join(os.path.dirname(__file__), 'val.csv'))
            test_df.to_csv(os.path.join(os.path.dirname(__file__), 'test.csv'))

        # Convert text to POS tags
        # Learning on one domain (Yahoo dataset) is not transferring well
        # to another domain (Onesource). However, POS should be more
        # universal (in English).
        print('Creating POS data')
        nlp = spacy.load('en_core_web_sm')
        train_df = train_df.assign(x=train_df.x.apply(lambda x: convert_text_to_pos(x, nlp)))
        val_df = val_df.assign(x=val_df.x.apply(lambda x: convert_text_to_pos(x, nlp)))
        test_df = test_df.assign(x=test_df.x.apply(lambda x: convert_text_to_pos(x, nlp)))
        train_df.to_csv(os.path.join(os.path.dirname(__file__), 'train_pos.csv'))
        val_df.to_csv(os.path.join(os.path.dirname(__file__), 'val_pos.csv'))
        test_df.to_csv(os.path.join(os.path.dirname(__file__), 'test_pos.csv'))

    word2vec_filename = os.path.join(os.path.dirname(__file__), 'word2vec.bin')
    constants['word2vec_filename'] = word2vec_filename
    if not os.path.exists(word2vec_filename):
        print('Training word2vec model using POS data')
        model = train_word2vec_model(np.concatenate((train_df.x.values, val_df.x.values)))
        model.save(word2vec_filename)

    n_classes = len(classes)
    batch_size = constants['batch_size']
    allow_soft_placement = constants['allow_soft_placement']
    log_device_placement = constants['log_device_placement']
    if constants['test']:
        print('\nTesting...')
        x_raw = test_df.x.values
        checkpoint_dir = constants['checkpoint_dir']
        vocab_path = os.path.join(checkpoint_dir, '..', 'vocab')
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x_raw)))
        y_test = test_df.y.values
        preds = test(x_test, batch_size, checkpoint_dir, allow_soft_placement, log_device_placement, y_test)
        save_eval_to_csv(x_raw, preds, checkpoint_dir)
    else:
        print('\nTraining...')
        x_train, y_train, x_val, y_val, vocab_processor = preprocess(train_df, val_df, n_classes)
        train(x_train, y_train, x_val, y_val, vocab_processor, model=None,
              learning_rate=constants['learning_rate'],
              n_checkpoints=constants['n_checkpoints'],
              keep_prob=constants['keep_prob'],
              batch_size=batch_size,
              n_epochs=constants['n_epochs'],
              evaluate_every=constants['evaluate_every'],
              checkpoint_every=constants['checkpoint_every'],
              allow_soft_placement=allow_soft_placement,
              log_device_placement=log_device_placement,
              constants=constants)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Question Detector')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--embedding-size', dest='embed_size', type=int, help='embedding size')
    parser.add_argument('--filter-sizes', dest='filter_sizes', type=str, help='comma-separated filter sizes')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--data-dir', dest='data_dir', type=str, help='relative path to data')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str,
                        help='checkpoint directory from training run')
    parser.add_argument('--word2vec-filename', dest='word2vec_filename', type=str,
                        help='path to word2vec embeddings file')
    parser.add_argument('--test', dest='test',
                        help='run eval on the test dataset using a fixed checkpoint', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict['filter_sizes']:
        args_dict['filter_sizes'] = [x for x in args_dict['filter_sizes'].split(',')]

    run(args_dict)
