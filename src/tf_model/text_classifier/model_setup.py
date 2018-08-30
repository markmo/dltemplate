import numpy as np
import tensorflow as tf


class TextModel(object):

    def __init__(self, embeddings, non_static, n_hidden, seq_len, max_pool_size,
                 n_classes, emb_dim, filter_sizes, n_filters, l2_reg_lambda=0.):
        self.input_x = tf.placeholder(tf.int32, [None, seq_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, emb_dim, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')
        l2_loss = tf.constant(0.)
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if non_static:
                w = tf.Variable(embeddings, name='W')
            else:
                w = tf.constant(embeddings, name='W')

            self.embedded_chars = tf.nn.embedding_lookup(w, self.input_x)
            emb = tf.expand_dims(self.embedded_chars, -1)

        pooled_concat = []
        reduced = np.int32(np.ceil(seq_len * 1. / max_pool_size))
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # zero-padded so the convolution output has dimension [batch, seq_len, emb_dim, n_channels]
                n_prev = (filter_size - 1) / 2
                n_post = filter_size - 1 - n_prev
                pad_prev = tf.concat([self.pad] * n_prev, 1)
                pad_post = tf.concat([self.pad] * n_post, 1)
                emb_pad = tf.concat([pad_prev, emb, pad_post], 1)
                filter_shape = [filter_size, emb_dim, 1, n_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[n_filters]), name='b')
                conv = tf.nn.conv2d(emb_pad, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
                                        padding='SAME', name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, n_filters])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(pooled_concat, 2)
        pooled_concat = tf.nn.dropout(pooled_concat, self.keep_prob)
        lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        inputs = [tf.squeeze(x, [1]) for x in tf.split(pooled_concat, num_or_size_splits=int(reduced), axis=1)]
        outputs, state = tf.nn.static_rnn(lstm_cell, inputs, initial_state=self._initial_state,
                                          sequence_length=self.real_len)

        # Collect the appropriate last words into variable output, shape [batch_size, emb_dim]
        output = outputs[0]
        with tf.variable_scope('output'):
            tf.get_variable_scope().reuse_variables()
            ones = tf.ones([1, n_hidden], tf.float32)
            for i in range(1, len(outputs)):
                indices = self.real_len < (i + 1)
                indices = tf.to_float(indices)
                indices = tf.expand_dims(indices, -1)
                mat = tf.matmul(indices, ones)
                output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1. - mat))

        with tf.name_scope('output'):
            self.w = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b')
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.w, b, name='scores')
            self.preds = tf.argmax(self.scores, 1, name='preds')

        with tf.name_scope('loss'):
            # only named arguments accepted
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_preds = tf.equal(self.preds, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, 'float'), name='accuracy')

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.preds, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))
