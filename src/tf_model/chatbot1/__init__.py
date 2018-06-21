from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
from common.load_data import load_word2vec_embeddings
from common.load_opensubs_data import read_opensubs_data
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf_model.chatbot1.model_setup import Seq2SeqModel
from tf_model.chatbot1.utils import evaluate_results, get_symbol_to_id_mappings, get_word_embeddings
from tf_model.chatbot1.utils import get_vocab, train, START_SYMBOL, END_SYMBOL


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    data = read_opensubs_data('data/opensubs/OpenSubtitles')
    conversation_steps = [(u.split(), r.split()) for u, r in data]
    embeddings = load_word2vec_embeddings(limit=10000)
    _, vocab, vocab_size = get_vocab(conversation_steps)
    x_train, x_test = train_test_split(conversation_steps, test_size=0.2, random_state=42)

    word2id, id2word = get_symbol_to_id_mappings(vocab)
    embeddings_size = constants['embeddings_size']
    word_embeddings = get_word_embeddings(embeddings, id2word, vocab_size, embeddings_size)

    tf.reset_default_graph()
    model = Seq2SeqModel(hidden_size=constants['n_hidden'],
                         vocab_size=vocab_size,
                         n_encoder_layers=constants['n_encoder_layers'],
                         n_decoder_layers=constants['n_decoder_layers'],
                         max_iter=constants['max_iter'],
                         start_symbol_id=word2id[START_SYMBOL],
                         end_symbol_id=word2id[END_SYMBOL],
                         word_embeddings=word_embeddings,
                         word2id=word2id,
                         id2word=id2word)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    all_ground_truth, all_model_predictions, invalid_number_prediction_counts = \
        train(sess, model, x_train, x_test, embeddings, word2id, id2word,
              n_epochs=constants['n_epochs'],
              batch_size=constants['batch_size'],
              max_len=constants['max_len'],
              learning_rate=constants['learning_rate'],
              dropout_keep_prob=constants['dropout_keep_prob'])

    evaluate_results(all_ground_truth, all_model_predictions, invalid_number_prediction_counts)

    model.predict(sess, 'What is your name')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run TF Chatbot1 model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', type=int, help='number hidden layers')
    parser.add_argument('--embeddings-size', dest='embeddings_size', type=int, help='embeddings size')
    parser.add_argument('--max-len', dest='max_len', type=int, help='maximum length of input sequence')
    args = parser.parse_args()

    run(vars(args))
