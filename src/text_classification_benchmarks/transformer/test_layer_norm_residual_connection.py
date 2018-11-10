import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import LayerNormResidualConnection
import time


def test_layer_norm_residual_connection():
    start = time.time()
    batch_size = 128
    seq_len = 1000
    d_model = 512
    x = tf.ones((batch_size, seq_len, d_model))
    y = x * 3 - 0.5
    layer_norm_residual_conn = LayerNormResidualConnection(x, y, 0, 'encoder')
    output = layer_norm_residual_conn.layer_norm_residual_connection()
    end = time.time()
    print('x:', x, ', y:', y, ', output:', output, ', latency:', (end - start))
