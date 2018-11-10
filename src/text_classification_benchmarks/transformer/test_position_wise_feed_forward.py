import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import PositionWiseFeedForward
import time


def _test_position_wise_feed_forward():
    start = time.time()
    x = tf.ones((8, 1000, 512))  # batch_size=8, seq_len=10
    layer_i = 0
    position_wise_ff = PositionWiseFeedForward(x, layer_i)
    output = position_wise_ff.position_wise_feed_forward()
    end = time.time()
    print('x:', x, ', output:', output, ', latency:', (end - start))
    return output


def test_graph():
    with tf.Session() as sess:
        result = _test_position_wise_feed_forward()
        sess.run(tf.global_variables_initializer())
        result_ = sess.run(result)
        print('result_:', result_)
