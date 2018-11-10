import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.

    Uses an embedding layer, followed by convolutional, max-pooling, and softmax layers.
    """
    def __init__(self, seq_len, n_classes, vocab_size, embed_size, filter_sizes, n_filters, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, seq_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Keeping track of L2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.w = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.w, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv_maxpool_%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embed_size, 1, n_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[n_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        n_filters_total = n_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, n_filters_total])

        # Add Dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            w = tf.get_variable('W', shape=[n_filters_total, n_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b')
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='scores')
            self.predictions = tf.argmax(self.scores, axis=1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, 'float'), name='accuracy')
