import numpy as np
import tensorflow as tf


class BiLSTMCRFModel(object):
    """
    This is more Lego-like than it used to be!
    """

    def __init__(self, vocab_size, n_tags, embedding_dim, n_hidden_rnn):
        self.__declare_placeholders()
        self.__build_layers(vocab_size, embedding_dim, n_hidden_rnn, n_tags)
        self.__compute_loss()
        self.__optimize()

    def __declare_placeholders(self):
        # Placeholders for input and ground truth output
        self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
        self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')

        # Placeholder for lengths of the sequences
        self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')

        # Placeholder for a dropout keep probability. If we don't feed
        # a value for this placeholder, it will be equal to 1.0.
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])

        # Placeholder for a learning rate (tf.float32)
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])

    def __build_layers(self, vocab_size, embedding_dim, n_hidden_rnn, n_tags):
        """
        Defines a Bidirectional LSTM architecture, and computes logits from inputs.

        :param vocab_size:
        :param embedding_dim:
        :param n_hidden_rnn:
        :param n_tags:
        :return:
        """
        initial_embedding_matrix = np.random.randn(vocab_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix = tf.Variable(initial_embedding_matrix, dtype=tf.float32, name='embedding_matrix')

        forward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn),
                                                     input_keep_prob=self.dropout_ph,
                                                     output_keep_prob=self.dropout_ph,
                                                     state_keep_prob=self.dropout_ph)

        backward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn),
                                                      input_keep_prob=self.dropout_ph,
                                                      output_keep_prob=self.dropout_ph,
                                                      state_keep_prob=self.dropout_ph)

        # Look up embeddings for self.input_batch
        # Shape: [batch_size, sequence_len, embedding_dim]
        embeddings = tf.nn.embedding_lookup(embedding_matrix, self.input_batch)

        # Pass embeddings through a Bidirectional Dynamic RNN
        # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn]
        (output_forward, output_backward), _ = \
            tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                            backward_cell,
                                            embeddings,
                                            dtype=tf.float32,
                                            sequence_length=self.lengths)

        output = tf.concat([output_forward, output_backward], axis=2)

        # Dense layer on top
        # Shape: [batch_size, sequence_len, n_tags].
        self.logits = tf.layers.dense(output, n_tags, activation=None)

    def __compute_loss(self):
        """

        :return:
        """
        log_likelihood, transition_params = \
            tf.contrib.crf.crf_log_likelihood(self.logits, self.ground_truth_tags, self.lengths)

        self.transition_params = transition_params
        self.loss = tf.reduce_mean(-log_likelihood)

    def __optimize(self):
        """
        Specifies the optimizer and train_op for the model.

        :return:
        """
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # Gradient clipping for self.grads_and_vars to eliminate exploding gradients.
        # Note that you need to apply this operation only for gradients,
        # self.grads_and_vars also contains variables
        clip_norm = tf.cast(1.0, tf.float32)
        self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_prob):
        feed_dict = {
            self.input_batch: x_batch,
            self.ground_truth_tags: y_batch,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_prob,
            self.lengths: lengths
        }
        session.run(self.train_op, feed_dict=feed_dict)

    def predict_for_batch(self, session, x_batch, lengths):
        """
        Instead of using softmax to decode scores into predictions, we're going
        to use a linear-chain CRF instead. The first method makes local choices.
        In other words, even if we capture some information from the context in
        our output thanks to the bi-LSTM, the tagging decision is still local.

        For instance, in "New York", the fact that we are tagging "York" as a
        location should help us to decide that "New" corresponds to the beginning
        of a location.

        To make the final predictions with the CRF, we have to use dynamic
        programming. Fortunately, there is a contributed package in TensorFlow
        that will do this for us. This function is pure Python at present.

        :param session:
        :param x_batch:
        :param lengths:
        :return:
        """
        feed_dict = {
            self.input_batch: x_batch,
            self.lengths: lengths
        }
        viterbi_sequences = []
        logits, transition_params = session.run([self.logits, self.transition_params], feed_dict=feed_dict)

        # iterate over the sentences because no batching in `viterbi_decode`
        for logit, length in zip(logits, lengths):
            logit = logit[:length]  # keep only the valid steps
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences
