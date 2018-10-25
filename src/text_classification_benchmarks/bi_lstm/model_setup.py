import tensorflow as tf


class BiLSTM(object):

    def __init__(self, n_classes, vocab_size, n_hidden, n_layers, l2_reg_lambda):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.l2_reg_lambda = l2_reg_lambda

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')

        self.l2_loss = tf.constant(0.)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[vocab_size, n_hidden], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # Input dropout
        self.inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        self.final_state = self.bi_lstm()

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[2 * n_hidden, n_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[n_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            self.logits = tf.matmul(self.final_state, softmax_w) + softmax_b

            preds = tf.nn.softmax(self.logits)
            self.preds = tf.argmax(preds, 1, name='preds')

        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()

            # L2 regularization for LSTM weights
            for v in tvars:
                if 'kernel' in v.name:
                    self.l2_loss += tf.nn.l2_loss(v)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_preds = tf.equal(self.preds, self.input_y)
            self.correct_count = tf.reduce_sum((tf.cast(correct_preds, tf.float32)))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

        self._initial_state_fw = None
        self._initial_state_bw = None

    def bi_lstm(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True,
                                          reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True,
                                          reuse=tf.get_variable_scope().reuse)

        # Add dropout to cell output
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] * self.n_layers, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] * self.n_layers, state_is_tuple=True)

        self._initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        self._initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        # Dynamic Bi-LSTM
        with tf.variable_scope('Bi-LSTM'):
            _, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=self.inputs,
                                                       initial_state_fw=self._initial_state_fw,
                                                       initial_state_bw=self._initial_state_bw,
                                                       sequence_length=self.seq_len)

        state_fw = state[0]
        state_bw = state[1]
        output = tf.concat([state_fw[self.n_layers - 1].h, state_bw[self.n_layers - 1].h], 1)

        return output
