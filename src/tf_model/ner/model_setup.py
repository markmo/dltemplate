import numpy as np
import tensorflow as tf


class BiLSTMModel(object):
    """
    This is more Lego-like than it used to be!
    """

    def __init__(self, vocab_size, n_tags, embedding_dim, n_hidden_rnn, pad_idx):
        self.__declare_placeholders()
        self.__build_layers(vocab_size, embedding_dim, n_hidden_rnn, n_tags)
        self.__compute_predictions()
        self.__compute_loss(n_tags, pad_idx)
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
        self.learning_rate_ph = tf.placeholder_with_default(1e4, shape=[])

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
        embedding_matrix_var = tf.Variable(initial_embedding_matrix, dtype=tf.float32, name='embedding_matrix')

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
        embeddings = tf.nn.embedding_lookup(embedding_matrix_var, self.input_batch)

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

    def __compute_predictions(self):
        """
        Transforms logits to probabilities and finds the most probable tags.

        :return:
        """
        # Create softmax function
        softmax_output = tf.nn.softmax(self.logits)

        # Use argmax to get the most probable tags
        self.predictions = tf.argmax(softmax_output, axis=-1)

    def __compute_loss(self, n_tags, pad_idx):
        """
        Computes masked cross-entropy loss with logits to ignore padding
        in loss calculations.

        :return:
        """
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                              labels=ground_truth_tags_one_hot)

        mask = tf.cast(tf.not_equal(self.input_batch, pad_idx), tf.float32)

        # Create loss function which doesn't operate with <PAD> tokens
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(loss_tensor, mask), axis=-1)
                                   / tf.reduce_sum(mask, axis=-1))

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
        feed_dict = {
            self.input_batch: x_batch,
            self.lengths: lengths
        }
        return session.run(self.predictions, feed_dict=feed_dict)
