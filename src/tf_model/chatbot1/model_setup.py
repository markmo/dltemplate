import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import optimize_loss
from tensorflow.contrib.rnn import MultiRNNCell, OutputProjectionWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode, sequence_loss
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper, TrainingHelper
from tf_model.chatbot1.utils import UNK_SYMBOL


class Seq2SeqModel(object):

    def __init__(self, hidden_size, vocab_size, n_encoder_layers, n_decoder_layers,
                 max_iter, start_symbol_id, end_symbol_id, word_embeddings, word2id, id2word):
        self.max_iter = max_iter
        self.word2id = word2id
        self.id2word = id2word
        self.__declare_placeholders()
        self.__build_input_encoder(n_encoder_layers, hidden_size, word_embeddings)
        self.__build_decoder(n_decoder_layers, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

        # Compute loss and back-propagate
        self.__compute_loss()
        self.__optimize()

        # Get predictions for evaluation
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

    def __build_input_encoder(self, n_encoder_layers, hidden_size, word_embeddings):
        # noinspection PyUnusedLocal
        with tf.variable_scope('input_encoder') as input_encoder_scope:
            # Specifies embeddings layer and embeds an input batch
            self.embeddings = tf.Variable(word_embeddings, dtype=tf.float32, name='embeddings')

            # Perform embeddings lookup for self.input_batch
            self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_batch)
            rnn_layers = []
            for i in range(n_encoder_layers):
                # Create GRUCell with dropout.
                cell = tf.nn.rnn_cell.GRUCell(hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_ph, dtype=tf.float32)
                rnn_layers.append(cell)

            encoder_cell = MultiRNNCell(rnn_layers)

            # Create RNN with the predefined cell.
            outputs, final_encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                             self.input_batch_embedded,
                                                             sequence_length=self.input_batch_lengths,
                                                             dtype=tf.float32)
            self.input_encoder_outputs = outputs
            self.final_encoder_state = final_encoder_state[-1]

    # noinspection PyUnusedLocal
    def __build_decoder(self, n_decoder_layers, hidden_size, vocab_size, max_iter,
                        start_symbol_id, end_symbol_id):
        # Use start symbols as the decoder inputs at the first time step
        batch_size = tf.shape(self.input_batch)[0]
        start_tokens = tf.fill([batch_size], start_symbol_id)
        ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

        # Use the embedding layer defined before to lookup embeddings for ground_truth_as_input
        self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings, ground_truth_as_input)

        # Create TrainingHelper for the train stage
        train_helper = TrainingHelper(self.ground_truth_embedded, self.ground_truth_lengths)

        # Create GreedyEmbeddingHelper for the inference stage
        infer_helper = GreedyEmbeddingHelper(self.embeddings, start_tokens, end_symbol_id)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                rnn_layers = []
                for i in range(n_decoder_layers):
                    # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
                    cell = tf.nn.rnn_cell.GRUCell(hidden_size, reuse=reuse)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_ph)
                    rnn_layers.append(cell)

                decoder_cell = MultiRNNCell(rnn_layers)

                # Create a projection wrapper
                decoder_cell = OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)

                # Create BasicDecoder, pass the defined cell, a helper, and initial state
                # The initial state should be equal to the final state of the encoder!
                initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                decoder = BasicDecoder(decoder_cell, helper, initial_state=initial_state)

                # The first returning argument of dynamic_decode contains two fields:
                #   * rnn_output (predicted logits)
                #   * sample_id (predictions)
                max_iters = tf.reduce_max(self.ground_truth_lengths)
                # max_iters = max_iter
                outputs, _, _ = dynamic_decode(decoder=decoder,
                                               maximum_iterations=max_iters,
                                               output_time_major=False,
                                               impute_finished=True)

                return outputs

        self.train_outputs = decode(train_helper, 'decode')
        self.infer_outputs = decode(infer_helper, 'decode', reuse=True)

    def __compute_loss(self):
        weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)
        self.loss = sequence_loss(self.train_outputs.rnn_output, self.ground_truth, weights)

    def __optimize(self):
        self.train_op = optimize_loss(self.loss,
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
        pred, loss, _ = sess.run([self.train_predictions, self.loss, self.train_op],
                                 feed_dict=feed_dict)
        return pred, loss

    def predict(self, sess, utterance):
        unk_id = self.word2id[UNK_SYMBOL]
        x = [[self.word2id[word] if word in self.word2id else unk_id for word in utterance.split()]]
        x = np.array(x)
        x_seq_len = np.array([len(x)])
        pred = self.predict_for_batch(sess, x, x_seq_len, y_seq_len=np.array([self.max_iter]))
        return ' '.join([self.id2word[i] for i in pred[0]])

    def predict_for_batch(self, sess, x, x_seq_len, y_seq_len):
        feed_dict = {
            self.input_batch: x,
            self.input_batch_lengths: x_seq_len,
            self.ground_truth_lengths: y_seq_len
        }
        pred = sess.run([self.infer_predictions], feed_dict=feed_dict)[0]
        return pred

    def predict_for_batch_with_loss(self, sess, x, x_seq_len, y, y_seq_len):
        feed_dict = {
            self.input_batch: x,
            self.input_batch_lengths: x_seq_len,
            self.ground_truth: y,
            self.ground_truth_lengths: y_seq_len
        }
        pred, loss = sess.run([self.infer_predictions, self.loss], feed_dict=feed_dict)
        return pred, loss
