from argparse import ArgumentParser
from common.load_data import load_names
from common.util import get_char_tokens, map_token_to_id, merge_dict, to_token_id_matrix
from IPython.display import clear_output
from keras_model.rnn.hyperparams import get_constants
from keras_model.rnn.model_setup import Layers, rnn_one_step
from keras import backend as ke
from keras.objectives import categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import tensorflow as tf


def generate_sample(sess, x_t, h_t, next_h, next_probs, tokens, max_len, seed_phrase=' '):
    """
    Generates text given a phrase of length at least SEQ_LENGTH.

    :param sess TensorFlow session
    :param x_t
    :param h_t
    :param next_h
    :param next_probs
    :param tokens
    :param seed_phrase:
    :param max_len: optional input to set the number of characters to predict
    :return:
    """
    n_tokens = len(tokens)
    token_to_id = map_token_to_id(tokens)
    x_sequence = [token_to_id[token] for token in seed_phrase]
    sess.run(tf.assign(h_t, h_t.initial_value))

    # feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        sess.run(tf.assign(h_t, next_h), {x_t: [ix]})

    # start generating
    for _ in range(max_len - len(seed_phrase)):
        x_probs, _ = sess.run([next_probs, tf.assign(h_t, next_h)], {x_t: [x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens, p=x_probs[0]))

    return ''.join([tokens[ix] for ix in x_sequence])


def run(constant_overwrites):
    names = load_names()
    constants = merge_dict(get_constants(), constant_overwrites)
    tokens = get_char_tokens(names)
    max_length = max(map(len, names))
    n_tokens = len(tokens)
    data = {
        'max_length': max_length,
        'n_tokens': n_tokens
    }
    model = Layers(data, constants)
    input_sequence = model.input_sequence()
    batch_size = tf.shape(input_sequence)[1]
    predicted_probs = []
    h_prev = tf.zeros([batch_size, constants['rnn_units']])  # initial hidden state
    for t in range(max_length):
        x_t = input_sequence[t]
        probs_next, h_next = rnn_one_step(x_t, h_prev, model)
        h_prev = h_next
        predicted_probs.append(probs_next)

    predicted_probs = tf.stack(predicted_probs)
    predictions_matrix = tf.reshape(predicted_probs[:-1], [-1, n_tokens])
    answers_matrix = tf.one_hot(tf.reshape(input_sequence[1:], [-1]), n_tokens)

    # Define loss as categorical cross-entropy. Mind that
    # predictions are probabilities and not logits!
    loss = tf.reduce_mean(categorical_crossentropy(answers_matrix, predictions_matrix))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    sess = ke.get_session()
    sess.run(tf.global_variables_initializer())
    history = []
    for i in range(constants['n_epochs']):
        batch = to_token_id_matrix(sample(names, 32), max_len=max_length)
        loss_i, _ = sess.run([loss, optimizer], {input_sequence: batch})
        history.append(loss_i)
        if (i + 1) % constants['n_report_steps'] == 0:
            clear_output()
            plt.plot(history, label='loss')
            plt.legend()
            plt.show()

    x_t = tf.placeholder('int32', (None,))
    h_t = tf.Variable(np.zeros([1, constants['rnn_units']], 'float32'))
    next_probs, next_h = rnn_one_step(x_t, h_t, model)

    for i in range(10):
        print(generate_sample(sess, x_t, h_t, next_h, next_probs, tokens, max_length))

    for i in range(10):
        print(generate_sample(sess, x_t, h_t, next_h, next_probs, tokens, max_length, seed_phrase='mark'))


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run Keras RNN')
    parser.add_argument('--epochs', dest='n_epochs', help='number epochs')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.set_defaults(retrain=False)
    args = parser.parse_args()
    run(vars(args))
