import math
from six.moves import xrange
import tensorflow as tf

# from https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and
    the memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be expressed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image.

    We use a geometric sequence of timescales starting with `min_timescale`
    and ending with `max_timescale`.  The number of different timescales is
    equal to `channels // (n * 2)`. For each timescale, we generate the two
    sinusoidal signals `sin(timestep/timescale)` and `cos(timestep/timescale)`.
    All of these sinusoids are concatenated in the channels dimension.

    :param x: shape [batch_size, d1...dn, channels]
    :param min_timescale: (float)
    :param max_timescale: (float)
    :return: a tensor with the same shape as x
    """
    static_shape = x.get_shape().as_list()
    n_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    n_timescales = channels // (n_dims * 2)
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale))
                               / (tf.to_float(n_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(n_timescales))
                                            * -log_timescale_increment)
    for dim in xrange(n_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * n_timescales
        postpad = channels - (dim + 1) * 2 * n_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)

        for _ in xrange(n_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)

        x += signal

    return x
