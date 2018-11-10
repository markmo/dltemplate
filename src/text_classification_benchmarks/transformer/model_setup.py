import tensorflow as tf
import time


class PositionWiseFeedForward(object):
    """
    Position-wise Feed-Forward Networks

    In addition to attention sub-layers, each layer in our encoder and decoder contains a fully
    connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.

    FFN(x) = max(0, xW1 + b1) W2 + b2

    While the linear transformations are the same across different positions, they use different
    parameters from layer to layer. Another way of describing this is as two convolutions with
    kernel size 1. The dimensionality of input and output is d_model=512, and the inner layer has
    dimensionality d_ff=2048.
    """
    def __init__(self, x, layer_i, d_model=300, d_ff=2048):
        """

        :param x: shape [batch_size, seq_len, d_model]
        :param layer_i:  layer index
        :param d_model:
        :param d_ff:
        """
        shape_list = x.get_shape().as_list()
        assert(len(shape_list) == 3)
        self.x = x
        self.layer_i = layer_i
        self.d_model = d_model
        self.d_ff = d_ff
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    def position_wise_feed_forward(self):
        # conv1
        input_x = tf.expand_dims(self.x, axis=3)  # [batch_size, seq_len, d_model, 1]
        # conv2d.input: [None, sent_len, embed_size, 1], filter=[filter_size, self.embed_size, 1, self.n_filters]
        # output with padding: [None, sent_len, 1, 1]
        output_conv1 = tf.layers.conv2d(input_x, filters=self.d_ff, kernel_size=[1, self.d_model], padding='VALID',
                                        name='conv1', kernel_initializer=self.initializer, activation=tf.nn.relu)
        output_conv1 = tf.transpose(output_conv1, [0, 1, 3, 2])
        # print('output_conv1:', output_conv1)

        # conv2
        output_conv2 = tf.layers.conv2d(output_conv1, filters=self.d_model, kernel_size=[1, self.d_ff],
                                        padding='VALID', name='conv2', kernel_initializer=self.initializer,
                                        activation=None)
        return tf.squeeze(output_conv2)  # [batch_size, seq_len, d_model]


class MultiHeadAttention(object):
    """
    Multi-head attention.

    Three kinds of usage:
    1. attention for encoder
    2. attention for decoder (need a mask to pay attention for only known position)
    3. attention as bridge between encoder and decoder
    """

    def __init__(self, q, ks, vs, d_model, d_k, d_v, seq_len, h,
                 model_type=None, is_training=None, mask=None, dropout_rate=0.1):
        # print('q:', q.get_shape().as_list(), 'ks:', ks.get_shape().as_list(), 'vs:', vs.get_shape().as_list())
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.h = h
        self.q = q
        self.ks = ks
        self.vs = vs
        self.model_type = model_type
        self.is_training = is_training
        self.mask = mask
        self.dropout_rate = dropout_rate

    def multi_head_attention(self):
        # 1. linearly project the queries, keys and values h times
        #    with different learned linear projections to (d_k, d_k, d_v) dimensions
        q_projected = tf.layers.dense(self.q, units=self.d_model)  # [batch_size, seq_len, d_model]
        ks_projected = tf.layers.dense(self.ks, units=self.d_model)  # [batch_size, seq_len, d_model]
        vs_projected = tf.layers.dense(self.vs, units=self.d_model)  # [batch_size, seq_len, d_model]

        # 2. scaled dot product attention for each projected version of Q, K, V
        # [batch_size, h, seq_len, d_k]
        dot_product = self.scaled_dot_product_attention_batch(q_projected, ks_projected, vs_projected)

        # 3. concatenated
        batch_size, h, length, d_k = dot_product.get_shape().as_list()
        dot_product = tf.reshape(dot_product, shape=(-1, length, self.d_model))

        # 4. linear projection
        return tf.layers.dense(dot_product, units=self.d_model)  # [batch_size, seq_len, d_model]

    def scaled_dot_product_attention_batch_(self, q, ks, vs):
        # 1. Split Q, K, V
        q_heads = tf.stack(tf.split(q, self.h, axis=2), axis=1)
        k_heads = tf.stack(tf.split(ks, self.h, axis=2), axis=1)
        v_heads = tf.stack(tf.split(vs, self.h, axis=2), axis=1)

        # 2. Calculate dot product
        dot_product = tf.multiply(q_heads, k_heads)
        dot_product *= (1.0 / tf.sqrt(tf.cast(self.d_model, tf.float32)))  # [batch_size, seq_len, d_k]
        dot_product = tf.reduce_sum(dot_product, axis=-1, keep_dims=True)  # [batch_size, seq_len, 1]

        # 3. Add mask if None
        if self.mask is not None:
            mask = tf.expand_dims(self.mask, axis=-1)  # [batch_size, seq_len, 1]
            mask = tf.expand_dims(mask, axis=1)  # [batch_size, 1, seq_len, 1]
            dot_product += mask  # [batch_size, h, seq_len, 1]

        # 4. Get probability
        prob = tf.nn.softmax(dot_product)  # [batch_size, h, seq_len, 1]

        # 5. Final output
        return tf.multiply(prob, v_heads)  # [batch_size, h, seq_len, d_k]

    def scaled_dot_product_attention_batch(self, q, ks, vs):
        """
        Scaled dot product attention. Implementation like tensor2tensor from Google.

        :param q: query, shape [batch_size, seq_len, d_model]
        :param ks: keys, shape [batch_size, seq_len, d_model]
        :param vs: values, shape [batch_size, seq_len, d_model]
        :return:
        """
        # print('q:', q.get_shape().as_list(), 'ks:', ks.get_shape().as_list(), 'vs:', vs.get_shape().as_list())
        # 1. Split Q, K, V
        q_heads = tf.stack(tf.split(q, self.h, axis=2), axis=1)  # [batch_size, h, seq_len, d_k]
        k_heads = tf.stack(tf.split(ks, self.h, axis=2), axis=1)  # [batch_size, h, seq_len, d_k]
        v_heads = tf.stack(tf.split(vs, self.h, axis=2), axis=1)  # [batch_size, h, seq_len, d_k]
        # print('v_heads:', v_heads.get_shape().as_list())

        # 2. Calculate dot product of Q, K
        dot_product = tf.matmul(q_heads, k_heads, transpose_b=True)  # [batch_size, h, seq_len, seq_len]
        # [batch_size, h, seq_len, seq_len]
        dot_product *= (1.0 / tf.sqrt(tf.cast(self.d_model, tf.float32)))

        # 3. Add mask if None
        if self.mask is not None:
            mask_expand = tf.expand_dims(tf.expand_dims(self.mask, axis=0), axis=0)  # [1, 1, seq_len, seq_len]
            dot_product += mask_expand  # [batch_size, h, seq_len, seq_len]

        # 4. Get probability
        weights = tf.nn.softmax(dot_product)  # [batch_size, h, seq_len, seq_len]

        # Dropout
        weights = tf.nn.dropout(weights, 1.0 - self.dropout_rate)  # [batch_size, h, seq_len, seq_len]

        # 5. Final output
        return tf.matmul(weights, v_heads)  # [batch_size, h, seq_len, d_model]


def get_mask(seq_len):
    lower_triangle = tf.matrix_band_part(tf.ones([seq_len, seq_len]), -1, 0)
    return 1e9 * (1.0 - lower_triangle)


class LayerNormResidualConnection(object):
    """
    We employ a residual connection around each of the two sub-layers, followed by layer normalization.
    That is, the output of each sub-layer is `LayerNorm(x + Sublayer(x))`, where `Sublayer(x)` is the
    function implemented by the sub-layer itself.
    """

    def __init__(self, x, y, layer_i, model_type, residual_dropout=0.1, use_residual_conn=True):
        self.x = x
        self.y = y
        self.layer_i = layer_i
        self.model_type = model_type
        self.residual_dropout = residual_dropout
        self.use_residual_conn = use_residual_conn

    def layer_norm_residual_connection(self):
        # if self.use_residual_conn:
        #     x_residual = self.residual_connection()
        #     x_layer_norm = self.layer_normalization(x_residual)
        # else:
        x_layer_norm = self.layer_normalization(self.x)

        return x_layer_norm

    def residual_connection(self):
        return self.x + tf.nn.dropout(self.y, 1.0 - self.residual_dropout)

    def layer_normalization(self, x):
        """
        Layer normalize the tensor x, averaging over the last dimension.

        :param x: shape [batch_size, seq_len, d_model]
        :return:
        """
        filter_size = x.get_shape()[-1]  # last dimension of x, e.g. 512
        with tf.variable_scope('layer_normalization_{}_{}'.format(self.layer_i, self.model_type)):
            # 1. Normalize input by using mean and variance according to last dimension
            mean = tf.reduce_mean(x, axis=-1, keep_dims=True)  # [batch_size, seq_len, 1]
            variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keep_dims=True)  # [batch_size, seq_len, 1]
            norm_x = (x - mean) * tf.rsqrt(variance + 1e-6)  # [batch_size, seq_len, d_model]

            # 2. Rescale normalized input
            scale = tf.get_variable('layer_norm_scale', [filter_size], initializer=tf.ones_initializer)  # [filter_size]
            bias = tf.get_variable('layer_norm_bias', [filter_size], initializer=tf.ones_initializer)  # [filter_size]
            return norm_x * scale + bias  # [batch_size, seq_len, d_model]


class BaseClass(object):

    def __init__(self, d_model, d_k, d_v, seq_len, h, batch_size,
                 n_layers=6, model_type='encoder', decoder_sent_len=None):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.h = h
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.model_type = model_type
        self.decoder_sent_len = decoder_sent_len
        self.initializer = None

    # noinspection PyMethodMayBeStatic
    def sub_layer_position_wise_feed_forward(self, x, layer_i, model_type):
        """

        :param x: shape [batch_size, seq_len, d_model]
        :param layer_i: layer index
        :param model_type:  ['encoder', 'decoder', 'encoder_decoder_attention']
        :return: shape [batch_size, seq_len, d_model]
        """
        with tf.variable_scope('sub_layer_position_wise_feed_forward_{}_{}'.format(model_type, layer_i)):
            position_wise_ff = PositionWiseFeedForward(x, layer_i)
            return position_wise_ff.position_wise_feed_forward()

    def sub_layer_multi_head_attention(self, layer_i, q, ks, model_type,
                                       mask=None, is_training=None, keep_prob=None):
        """
        Multi-head attention as sub layer

        TODO: should `keep_prob` use the instance variable

        :param layer_i: layer index
        :param q: shape [batch_size, seq_len, embed_size]
        :param ks: shape [batch_size, seq_len, embed_size]
        :param model_type: ['encoder', 'decoder', 'encoder_decoder_attention']
        :param mask: when using mask, illegal connection will be mask as huge negative value,
                     so possibility it will become zero.
        :param is_training:
        :param keep_prob:
        :return: shape [batch_size, seq_len, d_model]
        """
        with tf.variable_scope('base_model_sub_layer_multi_head_attention_{}_{}'.format(model_type, layer_i)):
            # if attention for encoder and decoder has different lengths, use:
            if model_type != 'encoder' and self.seq_len != self.decoder_sent_len:
                length = self.decoder_sent_len
            else:
                length = self.seq_len

            # length = self.seq_len
            # print('length:', length)

            # 1. Get V as learned parameters
            vs = tf.get_variable('vs', shape=(self.batch_size, length, self.d_model), initializer=self.initializer)

            # 2. Call function of multi-head attention to get result
            multi_head_attn = MultiHeadAttention(q, ks, vs, self.d_model, self.d_k, self.d_v, self.seq_len, self.h,
                                                 model_type=model_type, is_training=is_training, mask=mask,
                                                 dropout_rate=(1.0 - keep_prob))
            return multi_head_attn.multi_head_attention()  # [batch_size * seq_len, d_model]

    # noinspection PyMethodMayBeStatic
    def sub_layer_layer_norm_residual_connection(self, layer_input, layer_output, layer_i, model_type,
                                                 keep_prob=None, use_residual_conn=True):
        """

        :param layer_input: shape [batch_size, seq_len, d_model]
        :param layer_output: shape [batch_size, seq_len, d_model]
        :param layer_i:
        :param model_type:
        :param keep_prob:
        :param use_residual_conn:
        :return:
        """
        layer_norm_residual_conn = LayerNormResidualConnection(layer_input, layer_output, layer_i, model_type,
                                                               residual_dropout=(1.0 - keep_prob),
                                                               use_residual_conn=use_residual_conn)
        return layer_norm_residual_conn.layer_norm_residual_connection()  # [batch_size, seq_len, d_model]


class AttentionEncoderDecoder(BaseClass):
    """
    Attention connect encoder and decoder

    In 'encoder-decoder attention' layers, the queries come from the previous
    decoder layer, and the memory keys and values come from the output of the
    encoder. This allows every position in the decoder to attend over all
    positions in the input sequence. This mimics the typical encoder-decoder
    attention mechanisms in sequence-to-sequence models.
    """

    def __init__(self, d_model, d_k, d_v, seq_len, h, batch_size, q, ks, layer_i, decoder_sent_len,
                 model_type='attention', mask=None, keep_prob=None):
        super().__init__(d_model, d_k, d_v, seq_len, h, batch_size)
        self.q = q
        self.ks = ks
        self.layer_i = layer_i
        self.model_type = model_type
        self.decoder_sent_len = decoder_sent_len
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.mask = mask
        self.keep_prob = keep_prob

    def attention_encoder_decoder(self):
        """ Call multi-head attention function to perform this task. """
        return self.sub_layer_multi_head_attention(self.layer_i, self.q, self.ks, self.model_type,
                                                   mask=self.mask, keep_prob=self.keep_prob)


class Encoder(BaseClass):
    """
    Encoder for the Transformer.

    Six layers; each layer has two sub-layers:
        1. Multi-head self-attention mechanism
        2. Position-wise fully connected feed-forward network

    For each sub-layer, use LayerNorm(x + Sublayer(x))

    Input and last output dimension: d_model
    """

    def __init__(self, d_model, d_k, d_v, seq_len, h, batch_size, n_layers, q, ks,
                 model_type='encoder', mask=None, keep_prob=None, use_residual_conn=True):
        """

        :param d_model:
        :param d_k:
        :param d_v:
        :param seq_len:
        :param h:
        :param batch_size:
        :param n_layers:
        :param q: embedded words, shape [batch_size * seq_len, embed_size]
        :param ks:
        :param model_type:
        :param mask:
        :param keep_prob:
        :param use_residual_conn:
        """
        super().__init__(d_model, d_k, d_v, seq_len, h, batch_size, n_layers=n_layers)
        self.q = q
        self.ks = ks
        self.model_type = model_type
        self.mask = mask
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.keep_prob = keep_prob
        self.use_residual_conn = use_residual_conn

    def encoder(self):
        start = time.time()
        print('Encoding started')
        q, ks = self.q, self.ks
        for layer_i in range(self.n_layers):
            q, ks = self.encoder_single_layer(q, ks, layer_i)
            # print('{}. Q: {}, Ks: {}'.format(layer_i, q, ks))

        end = time.time()
        print('Encoding ended. Q: {}, Ks: {}, latency: {}'.format(q, ks, end - start))
        return q, ks

    def encoder_single_layer(self, q, ks, layer_i):
        """
        Single layer for encoder.

        Each layer has two sub-layers:
            1. Multi-head self-attention mechanism
            2. Position-wise fully connected feed-forward network

        For each sub-layer, use LayerNorm(x + Sublayer(x))

        Input and last output dimension: d_model

        :param q: shape [batch_size * seq_len, d_model]
        :param ks: shape [batch_size * seq_len, d_model]
        :param layer_i:
        :return: output, shape [batch_size * seq_len, d_model]
        """
        # 1.1 Multi-head self-attention mechanism
        # [batch_size, seq_len, d_model]
        multi_head_attention_output = self.sub_layer_multi_head_attention(layer_i, q, ks, self.model_type,
                                                                          mask=self.mask, keep_prob=self.keep_prob)
        # 1.2 Use LayerNorm(x + Sublayer(x)), all dimensions=512
        multi_head_attention_output = \
            self.sub_layer_layer_norm_residual_connection(ks, multi_head_attention_output, layer_i,
                                                          'encoder_multi_head_attention',
                                                          keep_prob=self.keep_prob,
                                                          use_residual_conn=self.use_residual_conn)

        # 2.1 Position-wise fully connected feed-forward network
        position_wise_feed_forward_output = \
            self.sub_layer_position_wise_feed_forward(multi_head_attention_output, layer_i, self.model_type)
        # 2.2 Use LayerNorm(x + Sublayer(x)), all dimensions=512
        position_wise_feed_forward_output = \
            self.sub_layer_layer_norm_residual_connection(multi_head_attention_output,
                                                          position_wise_feed_forward_output,
                                                          layer_i,
                                                          'encoder_position_wise_ff',
                                                          keep_prob=self.keep_prob)
        return position_wise_feed_forward_output, position_wise_feed_forward_output


class Decoder(BaseClass):
    """
    Decoder for the Transformer.

        1. The decoder is composed of a stack of N-6 identical layers
        2. In addition to the two sub-layers in each encoder layer, the decoder inserts
           a third sub-layer that performs multi-head attention over the output of the
           encoder stack.
        3. Similar to the encoder, residual connections are used around each of the sub-
           layers, followed by layer normalization. We also modify the self-attention
           sub-layer in the decoder stack to prevent positions from attending to
           subsequent positions. This masking, combined with the fact that the output
           embeddings are offset by one position, ensures that the predictions for
           position `i` can depend only on the known outputs at positions less than `i`.
    """

    def __init__(self, d_model, d_k, d_v, seq_len, h, batch_size, q, ks, k_v_encoder, decoder_sent_len,
                 n_layers=6, model_type='decoder', is_training=True, mask=None, keep_prob=None):
        super().__init__(d_model, d_k, d_v, seq_len, h, batch_size, n_layers=n_layers)
        self.q = q
        self.ks = ks
        self.k_v_encoder = k_v_encoder
        self.model_type = model_type
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.is_training = is_training
        self.decoder_sent_len = decoder_sent_len
        self.mask = mask
        self.keep_prob = keep_prob

    def decoder(self):
        start = time.time()
        print('Decoding started')
        q, ks = self.q, self.ks
        for layer_i in range(self.n_layers):
            q, ks = self.decoder_single_layer(q, ks, layer_i)

        end = time.time()
        print('Decoding ended. Q: {}, Ks: {}, latency: {}'.format(q, ks, end - start))
        return q, ks

    def decoder_single_layer(self, q, ks, layer_i):
        """
        Single layer for decoder. Each layer has three sub-layers:
            1. Multi-head self-attention (masked) mechanism
            2. Multi-head attention over encoder output
            3. Position-wise fully connected feed-forward network

        For each sub-layer, use `LayerNorm(x + Sublayer(x))`

        input and last output dimension: d_model

        mask is a list; length is `seq_len`; each element is a scalar,
        e.g. `[1, 1, 1, -1000000, -1000000, -1000000, ..., -1000000]`

        :param q: shape [batch_size, seq_len, d_model]
        :param ks: shape [batch_size, seq_len, d_model]
        :param layer_i: layer index
        :return: output, shape [batch_size * seq_len, d_model]
        """
        # 1.1 Masked multi-head self-attention mechanism
        # [batch_size * seq_len, d_model]
        multi_head_attention_output = self.sub_layer_multi_head_attention(layer_i, q, ks, self.model_type,
                                                                          is_training=self.is_training,
                                                                          mask=self.mask,
                                                                          keep_prob=self.keep_prob)
        # 1.2 Use `LayerNorm(x + Sublayer(x))`; all dimensions=512
        multi_head_attention_output = \
            self.sub_layer_layer_norm_residual_connection(ks, multi_head_attention_output, layer_i,
                                                          'decoder_multi_head_attention', keep_prob=self.keep_prob)

        # 2.1 Multi-head attention over encoder output
        # IMPORTANT!!! check two params below:
        #     1. Q should be from decoder
        #     2. Ks should the encoder output
        attention_enc_dec = AttentionEncoderDecoder(self.d_model, self.d_k, self.d_v, self.seq_len, self.h,
                                                    self.batch_size, multi_head_attention_output, self.k_v_encoder,
                                                    layer_i, self.decoder_sent_len, keep_prob=self.keep_prob)
        # 2.2 Use `LayerNorm(x + Sublayer(x))`, all dimensions=512
        attention_enc_dec_output = attention_enc_dec.attention_encoder_decoder()
        attention_enc_dec_output = \
            self.sub_layer_layer_norm_residual_connection(multi_head_attention_output, attention_enc_dec_output,
                                                          layer_i, 'decoder_attention_encoder_decoder',
                                                          keep_prob=self.keep_prob)

        # 3.1 Position-wise fully connected feed-forward network
        position_wise_feed_forward_output = self.sub_layer_position_wise_feed_forward(attention_enc_dec_output,
                                                                                      layer_i, self.model_type)
        # 3.2 Use `LayerNorm(x + Sublayer(x))`, all dimensions=512
        position_wise_feed_forward_output = \
            self.sub_layer_layer_norm_residual_connection(attention_enc_dec_output, position_wise_feed_forward_output,
                                                          layer_i, 'decoder_position_ff', keep_prob=self.keep_prob)
        return position_wise_feed_forward_output, position_wise_feed_forward_output


class Transformer(BaseClass):
    """
    Transformer. Perform sequence-to-sequence solely on attention mechanism.

    For more detail, see paper `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_

        1. Position embedding for encoder and decoder input
        2. Encoder with multi-head attention, position-wise feed forward
        3. Decoder with multi-head attention for decoder input, position-wise feed forward,
           multi-head attention between encoder and decoder

    Encoder. Six layers; each layer has two sub-layers:
        1. Multi-head self-attention mechanism
        2. Position-wise fully connected feed-forward network

        For each sub-layer, use `LayerNorm(x + Sublayer(x))`, all dimensions=512

    Decoder.
        1. The decoder is composed of a stack pf N=6 identical layers.
        2. In addition to the two sub-layers in each encoder layer, the decoder inserts
           a third sub-layer, which performs multi-head attention over the output of
           the encoder stack.
        3. Similar to the encoder, residual connections are used around each of the
           sub-layers, followed by layer normalization. We also modify the self-
           attention sub-layer in the decoder stack to prevent positions from attending
           to subsequent positions. This masking, combined with the fact that the
           output embeddings ae offset by one position, ensures that the predictions
           for position `i` can depend only on the known outputs at positions less than `i`.
    """

    # noinspection PyUnusedLocal
    def __init__(self, n_classes, learning_rate, batch_size, decay_steps, decay_rate, seq_len,
                 vocab_size, embed_size, d_model, d_k, d_v, h, n_layers, l2_lambda=0.0001,
                 decoder_sent_len=60, initializer=tf.random_normal_initializer(stddev=0.1),
                 clip_gradients=5.0, use_residual_conn=False, is_training=False):
        super().__init__(d_model, d_k, d_v, seq_len, h, batch_size, n_layers=n_layers)
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_size = d_model
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        # noinspection PyTypeChecker
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.learning_rate_ = None
        self.initializer = initializer
        self.decoder_sent_len = decoder_sent_len
        self.clip_gradients = clip_gradients
        self.l2_lambda = l2_lambda
        self.is_training = is_training
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.seq_len], name='input_x')
        # self.decoder_input = tf.placeholder(tf.int32, [self.batch_size, self.decoder_sent_len], name='decoder_input')
        # self.input_y_label = tf.placeholder(tf.int32, [self.batch_size, self.decoder_sent_len], name='input_y_label')
        self.input_y_label = tf.placeholder(tf.int32, [self.batch_size], name='input_y_label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.use_residual_conn = use_residual_conn
        self.embedding, self.embedding_label, self.w_projection, self.b_projection = self.instantiate_weights()
        self.logits = self.inference()  # logits shape [batch_size, decoder_dent_len, n_classes]
        # self.predictions = tf.argmax(self.logits, axis=2, name='predictions')
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
        correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y_label)
        # self.accuracy = tf.constant(0.5)  # fake accuracy!
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        if self.is_training:  # If not training, then no need to calculate loss and back-prop
            # self.loss_val = self.loss_seq2seq()
            self.loss_val = self.loss()
            self.train_op = self.train()

    def inference(self):
        """
        Building blocks:
            1. Encoder - 6 layers; each layer has 2 sub-layers
            2. Decoder - 6 layers; each layer has 3 sub-layers
        :return:
        """
        # 1. embedding for encoder and decoder inputs
        # 1.1 position embedding for encoder input
        input_x_embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)  # [None, seq_len, embed_size]
        input_x_embedded = tf.multiply(input_x_embedded, tf.sqrt(tf.cast(self.d_model, dtype=tf.float32)))
        input_mask = tf.get_variable('input_mask', [self.seq_len, 1], initializer=self.initializer)
        input_x_embedded = tf.add(input_x_embedded, input_mask)  # [None, seq_len, embed_size]

        # 1.2 position embedding for decoder input
        # [None, decoder_sent_len, embed_size]
        # decoder_input_embedded = tf.nn.embedding_lookup(self.embedding_label, self.decoder_input)
        # decoder_input_embedded = tf.multiply(decoder_input_embedded, tf.sqrt(tf.cast(self.d_model, dtype=tf.float32)))
        # decoder_input_mask = tf.get_variable('decoder_input_mask', [self.decoder_sent_len, 1],
        #                                      initializer=self.initializer)
        # decoder_input_embedded = tf.add(decoder_input_embedded, decoder_input_mask)

        # 2. Encoder
        encoder = Encoder(self.d_model, self.d_k, self.d_v, self.seq_len, self.h, self.batch_size, self.n_layers,
                          q=input_x_embedded, ks=input_x_embedded, keep_prob=self.keep_prob,
                          use_residual_conn=self.use_residual_conn)
        q_encoded, k_encoded = encoder.encoder()  # K_v_encoder

        # 3. Decoder
        # mask = self.get_mask(self.decoder_sent_len)
        # decoder = Decoder(self.d_model, self.d_k, self.d_v, self.seq_len, self.h, self.batch_size,
        #                   q=decoder_input_embedded, ks=decoder_input_embedded, k_v_encoder=k_encoded,
        #                   decoder_sent_len=self.decoder_sent_len, n_layers=self.n_layers,
        #                   is_training=self.is_training, mask=mask, keep_prob=self.keep_prob)
        # q_decoded, k_decoded = decoder.decoder()  # [batch_size, decoder_sent_len, d_model]
        # k_decoded = tf.reshape(k_decoded, shape=(-1, self.d_model))
        # with tf.variable_scope('output'):
        #     # logits shape [batch_size * decoder_sent_len, self.n_classes]
        #     logits = tf.matmul(k_decoded, self.w_projection) + self.b_projection
        #     logits = tf.reshape(logits, shape=(self.batch_size, self.decoder_sent_len, self.n_classes))

        q_encoded = tf.reshape(q_encoded, shape=(self.batch_size, -1))
        with tf.variable_scope('output'):
            logits = tf.matmul(q_encoded, self.w_projection) + self.b_projection

        return logits

    def loss(self):
        with tf.variable_scope('loss'):
            # losses shape [batch_size, decoder_sent_len]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                  if ('bias' not in v.name) and ('alpha' not in v.name)]) * self.l2_lambda
            return loss + l2_losses

    def loss_seq2seq(self):
        with tf.variable_scope('loss'):
            # losses shape [batch_size, decoder_sent_len]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits)
            loss_batch = tf.reduce_sum(losses, axis=1) / self.decoder_sent_len  # [batch_size]
            loss = tf.reduce_mean(loss_batch)
            l2_losses = tf.add_n([tf.nn.l2_loss(v)
                                  for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            return loss + l2_losses

    def train(self):
        """ Based on the loss, use SGD to update parameter. """
        self.learning_rate_ = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                         self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.learning_rate_,
                                                   optimizer='Adam',
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_weights(self):
        """ Define all weights here. """
        with tf.variable_scope('embedding_projection'):  # embedding matrix
            embedding = tf.get_variable('embedding', shape=[self.vocab_size, self.embed_size],
                                        initializer=self.initializer)  # [vocab_size, embed_size]
            embedding_label = tf.get_variable('embedding_label', shape=[self.n_classes, self.embed_size],
                                              dtype=tf.float32)
            # w_projection = tf.get_variable('w_projection', shape=[self.d_model, self.n_classes],
            #                                initializer=self.initializer)  # [embed_size, label_size]
            w_projection = tf.get_variable('w_projection', shape=[self.seq_len * self.d_model, self.n_classes],
                                           initializer=self.initializer)  # [embed_size, label_size]
            b_projection = tf.get_variable('b_projection', shape=[self.n_classes])
            return embedding, embedding_label, w_projection, b_projection

    # noinspection PyMethodMayBeStatic
    def get_mask(self, seq_len):
        lower_triangle = tf.matrix_band_part(tf.ones([seq_len, seq_len]), -1, 0)
        result = -1e9 * (1.0 - lower_triangle)
        return result
