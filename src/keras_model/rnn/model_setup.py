from keras.layers import Dense, Embedding
import tensorflow as tf


class Layers(object):

    def __init__(self, data, constants):
        self.data = data
        self.constants = constants

    def input_sequence(self):
        return tf.placeholder('int32', (self.data['max_length'], None))

    def embed_x(self):
        return Embedding(self.data['n_tokens'], self.constants['embedding_size'])

    def get_h_next(self):
        """
        Dense layer that maps input and previous state to new hidden state
        [x_t, h_t] -> h_t+1
        :return:
        """
        return Dense(self.constants['rnn_units'], activation='relu')

    def get_probs(self):
        """
        Dense layer that maps current hidden state to probabilities of characters
        [h_t+1] -> P(x_t+1|h_t+1)

        :return:
        """
        return Dense(self.data['n_tokens'], activation='softmax')


def rnn_one_step(x_t, h_t, model):
    """
    Recurrent neural network step that produces next state and output
    given previous input and state.

    This function is called repeatedly to produce the whole sequence.

    :param x_t: character id matrix
    :param h_t: previous hidden state
    :param model model layers
    :return: probabilities of next phoneme
    """
    # convert character id into embedding
    x_t_embed = model.embed_x()(tf.reshape(x_t, [-1, 1]))[:, 0]

    # concatenate x embedding and previous h state
    x_and_h = tf.concat([x_t_embed, h_t], axis=1)

    # compute next state given x_and_h
    h_next = model.get_h_next()(x_and_h)

    # get output probabilities for language model P(x_next|h_next)
    output_probs = model.get_probs()(h_next)

    return output_probs, h_next
