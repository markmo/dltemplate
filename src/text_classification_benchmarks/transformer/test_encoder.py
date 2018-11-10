import pytest
import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import Encoder, get_mask


@pytest.fixture
def params():
    d_model = 512
    d_k = 64
    d_v = 64
    seq_len = 50
    h = 8
    batch_size = 4 * 32
    n_layers = 6
    vocab_size = 1000
    embed_size = d_model
    keep_prob = 0.5
    tf.reset_default_graph()
    initializer = tf.random_normal_initializer(stddev=0.1)
    embedding = tf.get_variable('embedding_e', shape=[vocab_size, embed_size], initializer=initializer)
    decoder_input_x = tf.placeholder(tf.int32, [batch_size, seq_len], name='encoder_input_x')
    embedded_words = tf.nn.embedding_lookup(embedding, decoder_input_x)  # [batch_size * seq_len, embed_size]
    q = embedded_words  # [batch_size * seq_len, embed_size]
    ks = embedded_words  # [batch_size * seq_len, embed_size]
    mask = get_mask(seq_len)
    encoder = Encoder(d_model, d_k, d_v, seq_len, h, batch_size, n_layers, q, ks, mask=mask, keep_prob=keep_prob)
    return encoder, q, ks


def test_encoder_single_layer(params):
    encoder, q, ks = params
    layer_i = 0
    q_shape = q.get_shape().as_list()
    print('Q shape:', q_shape)
    output = encoder.encoder_single_layer(q, ks, layer_i)
    print(output)


def test_encoder(params):
    encoder, _, _ = params
    output = encoder.encoder()
    print(output)


def test_sub_layer_multi_head_attention(params):
    encoder, q, ks = params
    layer_i = 0
    keep_prob = 0.5
    output = encoder.sub_layer_multi_head_attention(layer_i, q, ks,
                                                    model_type='encoder',
                                                    keep_prob=keep_prob)
    print(output)


def test_position_wise_feed_forward(params):
    encoder, q, ks = params
    layer_i = 0
    keep_prob = 0.5
    multi_head_attention_output = encoder.sub_layer_multi_head_attention(layer_i, q, ks,
                                                                         model_type='encoder',
                                                                         keep_prob=keep_prob)
    # d1, d2, d3 = multi_head_attention_output.get_shape().as_list()
    # input_x = tf.reshape(multi_head_attention_output, shape=[-1, d3])
    # ERR: wrong shape for `input_x`
    # output = encoder.sub_layer_position_wise_feed_forward(input_x, layer_i, model_type='encoder')
    # output = tf.reshape(output, shape=(d1, d2, d3))
    output = encoder.sub_layer_position_wise_feed_forward(multi_head_attention_output, layer_i, model_type='encoder')
    print(output)
