import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tf_model.im2latex.model.attention_cell import AttentionCell
from tf_model.im2latex.model.attention_mechanism import AttentionMechanism
from tf_model.im2latex.model.beam_search_decoder_cell import BeamSearchDecoderCell
from tf_model.im2latex.model.dynamic_decode import dynamic_decode
from tf_model.im2latex.model.greedy_decoder_cell import GreedyDecoderCell


class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """

    def __init__(self, config, n_tok, id_end):
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._tiles = 1 if config.decoding == 'greedy' else config.beam_size

    def __call__(self, training, img, formula, dropout):
        """
        Decodes an image into a sequence of token

        :param training: (tf.placeholder) bool
        :param img: (tf.Tensor) encoded image, shape (m, h, w, c)
        :param formula: (tf.placeholder) shape (m, t)
        :param dropout:
        :return: pred_train: (tf.Tensor) shape (?, ?, vocab_size) logits of each class
                 pred_test: (structure)
                    - pred.test.logits, same as pred_train
                    - pred.test.ids, shape (?, config.max_length_formula)
        """
        dim_embeddings = self._config.attn_cell_config.get('dim_embeddings')
        e = tf.get_variable('E', initializer=embedding_initializer(),
                            shape=[self._n_tok, dim_embeddings], dtype=tf.float32)

        start_token = tf.get_variable('start_token', initializer=embedding_initializer(),
                                      shape=[dim_embeddings], dtype=tf.float32)

        batch_size = tf.shape(img)[0]

        # training
        with tf.variable_scope('attn_cell', reuse=False):
            embeddings = get_embeddings(formula, e, dim_embeddings, start_token, batch_size)
            attn_meca = AttentionMechanism(img, self._config.attn_cell_config['dim_e'])
            recu_cell = LSTMCell(self._config.attn_cell_config['n_hidden_units'])
            attn_cell = AttentionCell(recu_cell, attn_meca, dropout, self._config.attn_cell_config, self._n_tok)
            train_outputs, _ = tf.nn.dynamic_rnn(attn_cell, embeddings, initial_state=attn_cell.initial_state())

        # decoding
        with tf.variable_scope('attn_cell', reuse=True):
            attn_meca = AttentionMechanism(img=img, dim_e=self._config.attn_cell_config['dim_e'], tiles=self._tiles)
            recu_cell = LSTMCell(self._config.attn_cell_config['n_hidden_units'], reuse=True)
            attn_cell = AttentionCell(recu_cell, attn_meca, dropout, self._config.attn_cell_config, self._n_tok)
            if self._config.decoding == 'greedy':
                decoder_cell = GreedyDecoderCell(e, attn_cell, batch_size, start_token, self._id_end)
            elif self._config.decoding == 'beam_search':
                decoder_cell = BeamSearchDecoderCell(e, attn_cell, batch_size,
                                                     start_token, self._id_end,
                                                     self._config.beam_size,
                                                     self._config.div_gamma,
                                                     self._config.div_prob)
            else:
                raise NotImplementedError('Unknown decoding type {}'.format(self._config.decoding))

            test_outputs, _ = dynamic_decode(decoder_cell, self._config.max_length_formula + 1)

            return train_outputs, test_outputs


def get_embeddings(formula, e, dim, start_token, batch_size):
    """
    Returns the embedding of the n-1 first elements in the formula concat
    with the start token

    :param formula: (tf.placeholder) tf.uint32
    :param e: (tf.Variable) matrix
    :param dim: (int) dimension of embeddings
    :param start_token: (tf.Variable)
    :param batch_size: tf variable extracted from placeholder
    :return: embeddings_train: tensor
    """
    formula_ = tf.nn.embedding_lookup(e, formula)
    start_token_ = tf.reshape(start_token, [1, 1, dim])
    start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])
    embeddings = tf.concat([start_tokens, formula_[:, :-1, :]], axis=1)
    return embeddings


def embedding_initializer():
    """
    Returns initializer for embeddings

    :return:
    """

    # noinspection PyUnusedLocal
    def _initializer(shape, dtype, partition_info=None):
        e = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        e = tf.nn.l2_normalize(e, -1)
        return e

    return _initializer
