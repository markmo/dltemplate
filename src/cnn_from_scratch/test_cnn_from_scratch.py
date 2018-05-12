from cnn_from_scratch import conv_backward, conv_forward, conv_single_step
from cnn_from_scratch import create_mask_from_window, distribute_value, pool_backward, pool_forward, zero_pad
import numpy as np


def test_zero_pad():
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)

    assert x.shape == (4, 3, 3, 2), 'wrong shape'
    assert x_pad.shape == (4, 7, 7, 2), 'wrong shape'
    assert np.allclose(x[1, 1], [[0.90085595, -0.68372786], [-0.12289023, -0.93576943], [-0.26788808, 0.53035547]])
    assert np.array_equal(x_pad[1, 1], [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])


def test_conv_single_step():
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    w = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    z = conv_single_step(a_slice_prev, w, b)

    assert np.allclose(z, -23.1602122025)


def test_conv_forward():
    np.random.seed(1)
    a_prev = np.random.randn(10, 4, 4, 3)
    w = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    constants = {'pad': 2, 'stride': 1}
    z, cache_conv = conv_forward(a_prev, w, b, constants)

    assert np.allclose(np.mean(z), 0.155859324889)
    assert np.allclose(cache_conv[0][1][2][3], [-0.20075807, 0.18656139, 0.41005165])


def test_pool_forward():
    np.random.seed(1)
    a_prev = np.random.randn(2, 4, 4, 3)
    constants = {'stride': 1, 'filters': 4}
    a, _ = pool_forward(a_prev, constants, mode='max')
    assert np.allclose(a, [[[[1.74481176, 1.6924546, 2.10025514]]], [[[1.19891788, 1.51981682, 2.18557541]]]])
    a, _ = pool_forward(a_prev, constants, mode='average')
    assert np.allclose(a, [[[[-0.09498456, 0.11180064, -0.14263511]]], [[[-0.09525108, 0.28325018, 0.33035185]]]])


def test_conv_backward():
    np.random.seed(1)
    a_prev = np.random.randn(10, 4, 4, 3)
    w = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    constants = {'pad': 2, 'stride': 1}
    z, cache_conv = conv_forward(a_prev, w, b, constants)
    da, dw, db = conv_backward(z, cache_conv)

    assert np.allclose(np.mean(da), 9.60899067587)
    assert np.allclose(np.mean(dw), 10.5817412755)
    assert np.allclose(np.mean(db), 76.3710691956)


def test_create_mask_from_window():
    np.random.seed(1)
    x = np.random.randn(2, 3)
    mask = create_mask_from_window(x)

    assert np.allclose(x, [[1.62434536, -0.61175641, -0.52817175], [-1.07296862, 0.86540763, -2.3015387]])
    assert np.array_equal(mask, [[True, False, False], [False, False, False]])


def test_distribute_value():
    a = distribute_value(2, (2, 2))

    assert np.array_equal(a, [[0.5, 0.5], [0.5, 0.5]])


def test_pool_backward():
    np.random.seed(1)
    a_prev = np.random.randn(5, 5, 3, 2)
    constants = {'stride': 1, 'filters': 2}
    a, cache = pool_forward(a_prev, constants)
    da = np.random.randn(5, 4, 2, 2)
    da_prev = pool_backward(da, cache, mode='max')
    assert np.allclose(np.mean(da), 0.145713902729)
    assert np.allclose(da_prev[1, 1], [[0., 0.], [5.05844394, -1.68282702], [0., 0.]])
    da_prev = pool_backward(da, cache, mode='average')
    assert np.allclose(da_prev[1, 1], [[0.08485462, 0.2787552], [1.26461098, -0.25749373], [1.17975636, -0.53624893]])
