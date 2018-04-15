from homemade.classes import Dense, ReLU
from homemade.util import grad_softmax_crossentropy_with_logits, softmax_crossentropy_with_logits
from homemade.util_testing import eval_numerical_gradient
import numpy as np


def test_relu_gradients():
    inp = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
    layer = ReLU()
    grads = layer.backward(inp, np.ones([10, 32]) / (10 * 32))
    numeric_grads = eval_numerical_gradient(lambda x: layer.forward(x).mean(), x=inp)
    assert np.allclose(grads, numeric_grads, rtol=1e-3, atol=0), \
        'gradient returned by your layer does not match the numerically computed gradient'


def test_dense_layer():
    layer = Dense(128, 150)

    assert -0.05 < layer.weights.mean() < 0.05 and 1e-3 < layer.weights.std() < 1e-1, \
        "The initial weights must have zero mean and small variance. " \
        "If you know what you're doing, remove this assertion."
    assert -0.05 < layer.biases.mean() < 0.05, \
        "Biases must be zero mean. Ignore if you have a reason to do otherwise."

    # To test the outputs, we explicitly set weights with fixed values.
    layer = Dense(3, 4)
    inp = np.linspace(-1, 1, 2 * 3).reshape([2, 3])
    layer.weights = np.linspace(-1, 1, 3 * 4).reshape([3, 4])
    layer.biases = np.linspace(-1, 1, 4)

    assert np.allclose(layer.forward(inp), np.array([[0.07272727, 0.41212121, 0.75151515, 1.09090909],
                                                     [-0.90909091, 0.08484848, 1.07878788, 2.07272727]]))


def test_dense_grads():
    inp = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
    layer = Dense(32, 64, learning_rate=0)

    grads = layer.backward(inp, np.ones([10, 64]))
    numeric_grads = eval_numerical_gradient(lambda x: layer.forward(x).sum(), x=inp)

    assert np.allclose(grads, numeric_grads, rtol=1e-3, atol=0), 'input gradient does not match numeric grad'


def test_gradients_wrt_params():

    def compute_out_given_wb(w, b):
        layer = Dense(32, 64, learning_rate=1)
        layer.weights = np.array(w)
        layer.biases = np.array(b)
        inp = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
        return layer.forward(inp)

    def compute_grad_by_params(w, b):
        layer = Dense(32, 64, learning_rate=1)
        layer.weights = np.array(w)
        layer.biases = np.array(b)
        inp = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
        layer.backward(inp, np.ones([10, 64]) / 10.)
        return w - layer.weights, b - layer.biases

    weights, bias = np.random.randn(32, 64), np.linspace(-1, 1, 64)

    grad_w, grad_b = compute_grad_by_params(weights, bias)
    numeric_dw = eval_numerical_gradient(lambda w: compute_out_given_wb(w, bias).mean(0).sum(), weights)
    numeric_db = eval_numerical_gradient(lambda b: compute_out_given_wb(weights, b).mean(0).sum(), bias)

    assert np.allclose(grad_w, numeric_dw, rtol=1e-3, atol=0), 'weight gradient does not match numeric weight gradient'
    assert np.allclose(grad_b, numeric_db, rtol=1e-3, atol=0), 'bias gradient does not match numeric bias gradient'


def test_loss():
    logits = np.linspace(-1, 1, 500).reshape([50, 10])
    answers = np.arange(50) % 10

    softmax_crossentropy_with_logits(logits, answers)
    grads = grad_softmax_crossentropy_with_logits(logits, answers)
    numeric_grads = eval_numerical_gradient(lambda l: softmax_crossentropy_with_logits(l, answers).mean(), logits)

    assert np.allclose(grads, numeric_grads, rtol=1e-3, atol=0), \
        'The reference implementation has just failed. Someone has just changed the rules of math.'
