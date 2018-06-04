from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf_model.seq2seq.model_setup import Seq2SeqModel
from tf_model.seq2seq.util import evaluate_results, generate_equations, get_symbol_to_id_mappings, train


def run(constant_overwrites):
    allowed_operators = ['+', '-']
    dataset_size = 100000
    data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=9999)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    word2id, id2word = get_symbol_to_id_mappings()

    # Special symbols
    start_symbol = '^'  # indicate the beginning of the decoding procedure
    end_symbol = '$'  # indicate the end of a string, both for input and output sequences
    # padding_symbol = '#'  # a padding character to make lengths of all strings equal within one training batch

    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    model = Seq2SeqModel(vocab_size=len(word2id),
                         embeddings_size=constants['embeddings_size'],
                         hidden_size=constants['n_hidden'],
                         max_iter=constants['max_iter'],
                         start_symbol_id=word2id[start_symbol],
                         end_symbol_id=word2id[end_symbol])

    sess = tf.Session()
    all_ground_truth, all_model_predictions, invalid_number_prediction_counts = \
        train(sess, model, train_set, test_set, word2id, id2word, constants)

    evaluate_results(all_ground_truth, all_model_predictions, invalid_number_prediction_counts)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run TF Seq2Seq model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', type=int, help='number hidden layers')
    args = parser.parse_args()

    run(vars(args))
