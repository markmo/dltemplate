import collections
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


AttentionState = collections.namedtuple('AttentionState', ('cell_state', 'o'))


class AttentionCell(RNNCell):

    def __init__(self, cell, attention_mechanism,
                 dropout_keep_prob, constants,
                 n_proj, dtype=tf.float32):
        """

        :param cell: (RNNCell)
        :param attention_mechanism: (AttentionMechanism)
        :param dropout_keep_prob: (tf.float)
        :param constants: (dict) hyperparams
        :param n_proj:
        :param dtype:
        """
        super().__init__()
        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._dropout_keep_prob = dropout_keep_prob

        # hyperparameters and shapes
        self._n_channels = attention_mechanism.n_channels
        self._dim_e = constants['dim_e']
        self._dim_output = constants['dim_output']
        self._n_hidden_units = constants['n_hidden_units']
        self._dim_embeddings = constants['dim_embeddings']
        self._n_proj = n_proj
        self._dtype = dtype

        self._state_size = AttentionState(cell.state_size, self._dim_output)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._n_proj

    @property
    def output_dtype(self):
        return self._dtype

    def initial_state(self):
        """
        Initial state of the LSTM

        :return:
        """
        initial_cell_state = self._attention_mechanism.initial_cell_state(self._cell)
        initial_o = self._attention_mechanism.initial_state('o', self._dim_output)
        return AttentionState(initial_cell_state, initial_o)

    def step(self, embedding, attn_cell_state):
        prev_cell_state, o = attn_cell_state
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute new h
            x = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self._cell(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout_keep_prob)

            # compute attention
            c = self._attention_mechanism.context(new_h)

            # compute o
            w_c = tf.get_variable('output_w_c', dtype=tf.float32,
                                  shape=(self._n_channels, self._dim_output))

            w_h = tf.get_variable('output_w_h', dtype=tf.float32,
                                  shape=(self._n_hidden_units, self._dim_output))

            new_o = tf.tanh(tf.matmul(new_h, w_h) + tf.matmul(c, w_c))
            new_o = tf.nn.dropout(new_o, self._dropout_keep_prob)

            w_o = tf.get_variable('w_o', dtype=tf.float32,
                                  shape=(self._dim_output, self._n_proj))

            logits = tf.matmul(new_o, w_o)
            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state

    # noinspection PyMethodOverriding
    def __call__(self, inputs, state):
        """

        :param inputs: the embedding of the previous word for training only
        :param state: (AttentionState) (h, o) where h is the hidden state and
                      o is the output vector used to make the prediction of
                      the previous word
        :return:
        """
        new_output, new_state = self.step(inputs, state)
        return new_output, new_state
