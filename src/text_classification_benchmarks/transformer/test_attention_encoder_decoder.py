import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import AttentionEncoderDecoder
import time


def test_attention_encoder_decoder():
    start = time.time()
    d_model = 512
    d_k = 64
    d_v = 64
    seq_len = 600
    decoder_sent_len = 600
    h = 8
    batch_size = 128
    initializer = tf.random_normal_initializer(stddev=0.1)
    vocab_size = 1000
    embed_size = d_model
    embedding = tf.get_variable('embedding', shape=[vocab_size, embed_size], initializer=initializer)
    input_x = tf.placeholder(tf.int32, [batch_size, decoder_sent_len], name='input_x')
    embedded_words = tf.nn.embedding_lookup(embedding, input_x)  # [batch_size * seq_len, embed_size]
    q = embedded_words
    ks = tf.ones((batch_size, seq_len, embed_size), dtype=tf.float32)  # [batch_size * seq_len, embed_size]
    layer_i = 0
    mask = None
    keep_prob = 0.5
    attn_bw_enc_dec = AttentionEncoderDecoder(d_model, d_k, d_v, seq_len, h, batch_size, q, ks,
                                              layer_i, decoder_sent_len, mask=mask, keep_prob=keep_prob)
    output = attn_bw_enc_dec.attention_encoder_decoder()
    end = time.time()
    print('embedded_words:', embedded_words, 'attention_output:', output, 'latency:', end - start)
