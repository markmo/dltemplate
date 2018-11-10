import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import get_mask, MultiHeadAttention
import time


def test_vectorized_multi_head_attention_for_sentence():
    """ Vectorized implementation of multi-head attention for sentences batch. """
    start = time.time()

    # 1. Set parameters
    layer_i = 0
    d_model = 512
    d_k = 64
    d_v = 64
    seq_len = 1000
    h = 8
    batch_size = 128
    initializer = tf.random_normal_initializer(stddev=0.1)

    # 2. Set Q, K, V
    vocab_size = 1000
    embed_size = d_model
    model_type = 'decoder'
    embedding = tf.get_variable('embedding', shape=[vocab_size, embed_size], initializer=initializer)
    input_x = tf.placeholder(tf.int32, [batch_size, seq_len], name='input_x')
    embedded_words = tf.nn.embedding_lookup(embedding, input_x)  # [batch_size, seq_len, embed_size]
    mask = get_mask(seq_len)
    with tf.variable_scope('query_at_each_sentence_{}'.format(layer_i)):
        q = embedded_words  # [batch_size * seq_len, embed_size]
        ks = embedded_words  # [batch_size * seq_len, embed_size]
        vs = tf.get_variable('vs_original', shape=embedded_words.get_shape().as_list(),
                             initializer=initializer)  # [batch_size, seq_len, embed_size]

        # 3. Call method to get result
        multi_head_attn = MultiHeadAttention(q, ks, vs, d_model, d_k, d_v, seq_len, h,
                                             model_type=model_type, mask=mask)
        encoder_output = multi_head_attn.multi_head_attention()  # [seq_len, d_model]
        encoder_output = tf.reshape(encoder_output, shape=(batch_size, seq_len, d_model))

    end = time.time()
    print('input_x:', input_x)
    print('encoder_output:', encoder_output, ', latency:', (end - start))
