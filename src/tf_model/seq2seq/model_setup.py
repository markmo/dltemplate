import tensorflow as tf


class Seq2SeqModel(object):

    def __init__(self, vocab_size, embeddings_size, hidden_size,
                 max_iter, start_symbol_id, end_symbol_id):
        self.__declare_placeholders()
        self.__create_embeddings(vocab_size, embeddings_size)
        self.__build_encoder(hidden_size)
        self.__build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

        # Compute loss and back-propagate.
        self.__compute_loss()
        self.__optimize()

        # Get predictions for evaluation.
        self.train_predictions = self.train_outputs.sample_id
        self.infer_predictions = self.infer_outputs.sample_id

    def __declare_placeholders(self):
        # Placeholders for input and its actual lengths
        # sequences of sentences, shape [batch_size, max_sequence_len_in_batch]
        self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
        # lengths of unpadded sequences, shape [batch_size]
        self.input_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_batch_lengths')

        # Placeholders for ground-truth and its actual lengths
        # sequences of ground truth, shape [batch_size, max_sequence_len_in_batch]
        self.ground_truth = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ground_truth')
        # lengths of unpadded ground-truth sequences, shape [batch_size]
        self.ground_truth_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='ground_truth_lengths')

        # dropout keep probability, default value of 1.0
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
        self.learning_rate_ph = tf.placeholder(shape=[], dtype=tf.float32)

    def __create_embeddings(self, vocab_size, embeddings_size):
        # Specifies embeddings layer and embeds an input batch
        random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
        # Since we use the same vocabulary for input and output, we need only one such matrix.
        # For tasks with different vocabularies, there would be multiple embedding layers.
        self.embeddings = tf.Variable(random_initializer, dtype=tf.float32, name='embeddings')

        # Perform embeddings lookup for self.input_batch
        self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_batch)

    def __build_encoder(self, hidden_size):
        """
        Specifies encoder architecture and computes its output.

        It encodes an input sequence to a real-valued vector. Input of this RNN is an embedded
        input batch. Since sentences in the same batch could have different actual lengths, we
        also provide input lengths to avoid unnecessary computations. The final encoder state
        will be passed to the second RNN (decoder).

        :param hidden_size:
        :return:
        """
        # Create GRUCell with dropout.
        encoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size),
                                                     input_keep_prob=self.dropout_ph,
                                                     dtype=tf.float32)

        # Create RNN with the predefined cell.
        _, self.final_encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                        self.input_batch_embedded,
                                                        sequence_length=self.input_batch_lengths,
                                                        dtype=tf.float32)

    def __build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
        """
        Specifies decoder architecture and computes the output.

        Generates the output sequence. In this simple seq2seq architecture, the input sequence
        is provided to the decoder only as the final state of the encoder. Obviously, it is a
        bottleneck and Attention techniques can help to overcome it. So far, we do not need
        them to make our calculator work, but this would be a necessary ingredient for more
        advanced tasks.

        During training, the decoder also uses information about the true output. It is fed in
        as input, symbol by symbol. However, during the prediction stage (called inference),
        the decoder can use only its own generated output from the previous step to feed it to
        the next step. Because of this difference (training vs inference), we create two
        distinct instances, which will serve for the described scenarios.

        Uses helpers for:
          * training: feeding ground truth
          * inference: feeding generated output

        As a result, self.train_outputs and self.infer_outputs are created.
        Each of them contains two fields:
          * rnn_output (predicted logits)
          * sample_id (predictions)

        :param hidden_size:
        :param vocab_size:
        :param max_iter:
        :param start_symbol_id:
        :param end_symbol_id:
        :return:
        """
        # Use start symbols as the decoder inputs at the first time step
        batch_size = tf.shape(self.input_batch)[0]
        start_tokens = tf.fill([batch_size], start_symbol_id)
        ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

        # Use the embedding layer defined before to lookup embeddings for ground _truth_as_input
        self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings, ground_truth_as_input)

        # Create TrainingHelper for the train stage
        train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded,
                                                         self.ground_truth_lengths)

        # Create GreedyEmbeddingHelper for the inference stage
        # You should provide the embedding layer, start_tokens and index of the end symbol
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_symbol_id)

        def decode(helper, scope, reuse=None):
            """
            Creates decoder and return the results of the decoding with a given helper.

            :param helper:
            :param scope:
            :param reuse:
            :return:
            """
            with tf.variable_scope(scope, reuse=reuse):
                # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size,
                                                                                    reuse=reuse),
                                                             input_keep_prob=self.dropout_ph)

                # Create a projection wrapper
                decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size,
                                                                      reuse=reuse)

                # Create BasicDecoder, pass the defined cell, a helper, and initial state
                # The initial state should be equal to the final state of the encoder!
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                          initial_state=self.final_encoder_state)

                # The first returning argument of dynamic_decode contains two fields:
                #   * rnn_output (predicted logits)
                #   * sample_id (predictions)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                  maximum_iterations=max_iter,
                                                                  output_time_major=False,
                                                                  impute_finished=True)

                return outputs

        self.train_outputs = decode(train_helper, 'decode')
        self.infer_outputs = decode(infer_helper, 'decode', reuse=True)

    def __compute_loss(self):
        """
        Computes sequence loss (masked cross-entropy loss with logits).

        `sequence_loss` is a weighted cross-entropy loss for a sequence of logits.

        Note that we do not want to take into account loss terms from padding symbols,
        so we mask this out using weights.

        :return:
        """
        weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.train_outputs.rnn_output, self.ground_truth, weights)

    def __optimize(self):
        """
        Specifies train_op that optimizes self.loss.

        :return:
        """
        self.train_op = tf.contrib.layers.optimize_loss(self.loss,
                                                        global_step=tf.train.get_global_step(),
                                                        learning_rate=self.learning_rate_ph,
                                                        optimizer='Adam',
                                                        clip_gradients=1.0)

    def train_on_batch(self, sess, x, x_seq_len, y, y_seq_len, learning_rate, dropout_keep_prob):
        feed_dict = {
            self.input_batch: x,
            self.input_batch_lengths: x_seq_len,
            self.ground_truth: y,
            self.ground_truth_lengths: y_seq_len,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_prob
        }
        pred, loss, _ = sess.run([self.train_predictions, self.loss, self.train_op], feed_dict=feed_dict)
        return pred, loss

    def predict_for_batch(self, sess, x, x_seq_len):
        """
        We implement two prediction functions: `predict_for_batch` and `predict_for_batch_with_loss`.
        The first one only predicts output for some input sequence, while the second one can compute
        loss, as we also provide ground-truth values. Both these functions might be useful. The first
        can be used for predicting only, and the second is helpful for validating results on
        validation and test data during training.

        :param sess:
        :param x:
        :param x_seq_len:
        :return:
        """
        feed_dict = {
            self.input_batch: x,
            self.input_batch_lengths: x_seq_len
        }
        pred = sess.run([self.infer_predictions], feed_dict=feed_dict)[0]
        return pred

    def predict_for_batch_with_loss(self, sess, x, x_seq_len, y, y_seq_len):
        feed_dict = {
            self.input_batch: x,
            self.input_batch_lengths: x_seq_len,
            self.ground_truth: y,
            self.ground_truth_lengths: y_seq_len,
        }
        pred, loss = sess.run([self.infer_predictions, self.loss], feed_dict=feed_dict)
        return pred, loss
