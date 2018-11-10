import pytest
import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import Decoder, get_mask


@pytest.fixture
def params():
    d_model = 512
    d_k = 64
    d_v = 64
    seq_len = 6
    decoder_sent_len = 6
    h = 8
    batch_size = 4 * 32
    n_layers = 6
    vocab_size = 1000
    embed_size = d_model
    keep_prob = 0.5
    tf.reset_default_graph()
    initializer = tf.random_normal_initializer(stddev=0.1)
    embedding = tf.get_variable('embedding_d', shape=[vocab_size, embed_size], initializer=initializer)
    decoder_input_x = tf.placeholder(tf.int32, [batch_size, decoder_sent_len], name='decoder_input_x')
    decoder_input_embedding = tf.nn.embedding_lookup(embedding, decoder_input_x)  # [batch_size * seq_len, embed_size]
    q = tf.placeholder(tf.float32, [batch_size, seq_len, d_model], name='input_x')
    ks = decoder_input_embedding
    k_v_encoder = tf.get_variable('v_variable', shape=[batch_size, decoder_sent_len, d_model], initializer=initializer)
    mask = get_mask(decoder_sent_len)  # seq_len
    decoder = Decoder(d_model, d_k, d_v, seq_len, h, batch_size, q, ks, k_v_encoder, decoder_sent_len,
                      n_layers=n_layers, mask=mask, keep_prob=keep_prob)
    return decoder, q, ks


def test_decoder_single_layer(params):
    decoder, q, ks = params
    layer_i = 0
    q_shape = q.get_shape().as_list()
    print('Q shape:', q_shape)
    # seq_len_unfold = q_shape[0]
    # mask = tf.ones(seq_len_unfold)
    # output = decoder.decoder_single_layer(q, ks, layer_i, mask)
    output = decoder.decoder_single_layer(q, ks, layer_i)
    print(output)


def test_decoder(params):
    decoder, _, _ = params
    output = decoder.decoder()
    print(output)
