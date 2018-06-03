from argparse import ArgumentParser
from common.load_data import load_twitter_entities_dataset
from common.util import merge_dict
import tensorflow as tf
from tf_model.ner.hyperparams import get_constants
from tf_model.ner.model_setup import BiLSTMModel
from tf_model.ner.util import build_dict, eval_conll, train

PAD_TOKEN = '<PAD>'


def run(constant_overwrites):
    tokens_train, tags_train, tokens_val, tags_val, tokens_test, tags_test = load_twitter_entities_dataset()
    special_tokens = ['<UNK>', '<PAD>']
    special_tags = ['O']
    tok2idx, idx2tok = build_dict(tokens_train + tokens_val, special_tokens)
    tag2idx, idx2tag = build_dict(tags_train, special_tags)

    constants = merge_dict(get_constants(), constant_overwrites)
    vocab_size = len(tok2idx.keys())
    n_tags = len(tag2idx.keys())
    pad_idx = tok2idx[PAD_TOKEN]

    print('vocab_size:', vocab_size)
    print('n_tags:', n_tags)
    print('pad_idx:', pad_idx)

    tf.reset_default_graph()
    model = BiLSTMModel(vocab_size, n_tags, constants['embedding_dim'], constants['n_hidden'], pad_idx)

    sess = tf.Session()
    train(model, sess, tokens_train, tags_train, tokens_val, tags_val, tok2idx, tag2idx, idx2tok, idx2tag, constants)

    print('-' * 20 + ' Train set quality: ' + '-' * 20)
    eval_conll(model, sess, tokens_train, tags_train, tok2idx, tag2idx, idx2tok, idx2tag)

    print('-' * 20 + ' Validation set quality: ' + '-' * 20)
    eval_conll(model, sess, tokens_val, tags_val, tok2idx, tag2idx, idx2tok, idx2tag)

    print('-' * 20 + ' Test set quality: ' + '-' * 20)
    eval_conll(model, sess, tokens_test, tags_test, tok2idx, tag2idx, idx2tok, idx2tag)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run TF Bi-LSTM NER model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', type=int, help='number hidden layers')
    args = parser.parse_args()

    run(vars(args))
