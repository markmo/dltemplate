import numpy as np
from sonnet_model.cnn.model_setup import MyCNN
import tensorflow as tf


def test_initialize_parameters(capsys):
    np.random.seed(1)
    tf.reset_default_graph()
    with capsys.disabled():
        with tf.Session() as sess:
            model = MyCNN()
            x_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
            _, parameters = model(x_placeholder)
            sess.run(tf.global_variables_initializer())
            print('w1:', parameters['w1'].eval()[1, 1, 1])
            assert np.allclose(parameters['w1'].eval()[1, 1, 1],
                               [0.00131723, 0.14176141, -0.04434952, 0.09197326,
                                0.14984085, -0.03514394, -0.06847463, 0.05245192])
            assert np.allclose(parameters['w2'].eval()[1, 1, 1],
                               [-0.08566415, 0.17750949, 0.11974221, 0.16773748,
                                -0.0830943, -0.08058, -0.00577033, -0.14643836,
                                0.24162132, -0.05857408, -0.19055021, 0.1345228,
                                -0.22779644, -0.1601823, -0.16117483, -0.10286498])


def test_forward_propagation(capsys):
    np.random.seed(1)
    tf.reset_default_graph()
    with capsys.disabled():
        with tf.Session() as sess:
            model = MyCNN()
            x_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
            y_placeholder = tf.placeholder(tf.float32, [None, 6])
            z3, _ = model(x_placeholder)
            sess.run(tf.global_variables_initializer())
            a = sess.run(z3, {x_placeholder: np.random.randn(2, 64, 64, 3), y_placeholder: np.random.randn(2, 6)})
            assert np.allclose(a, [[-0.44670227, -1.57208765, -1.53049231, -2.31013036, -1.29104376, 0.46852064],
                                   [-0.17601591, -1.57972014, -1.4737016, -2.61672091, -1.00810647, 0.5747785]])
