import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops, variable_scope as vs


def circular_convolution(v, k):
    """

    :param v: a 1D Tensor (vector)
    :param k: a 1D Tensor (kernel)
    :return:
    """
    size = int(v.get_shape()[0])
    kernel_size = int(k.get_shape()[0])
    kernel_shift = int(math.floor(kernel_size / 2.))

    def loop(idx):
        if idx < 0:
            return size + idx

        if idx >= size:
            return idx - size

        return idx

    kernels = []
    for i in range(size):
        indices = [loop(i + j) for j in range(kernel_shift, -kernel_shift-1, -1)]
        v_ = tf.gather(v, indices)
        kernels.append(tf.reduce_sum(v_ * k, 0))

    return tf.dynamic_stitch([i for i in range(size)], kernels)


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    :param args: a 2D Tensor or a list of 2D, batch x n, Tensors
    :param output_size: (int) second dimension of W[i]
    :param bias: (bool) whether to add a bias term or not
    :param bias_start: starting value to initialize the bias; 0 by default.
    :param scope: VariableScope for the created subgraph; defaults to "Linear".
    :return: A 2D Tensor with shape [batch x output_size] equal to
             sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    :raise ValueError: if some of the arguments have unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1
    total_arg_size = 0
    shapes = []
    for arg in args:
        # noinspection PyBroadException
        try:
            shapes.append(arg.get_shape().as_list())
        except Exception:
            shapes.append(arg.shape)

    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
            total_arg_size += shape[0]
        else:
            total_arg_size += shape[1]

    with vs.variable_scope(scope or 'Linear'):
        matrix = vs.get_variable('Matrix', [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)

        if not bias:
            return res

        bias_term = vs.get_variable('Bias', [output_size],
                                    initializer=init_ops.constant_initializer(bias_start))

    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term


# noinspection PyPep8Naming
def Linear(input_, output_size, stddev=0.5, is_range=False, squeeze=False, name=None, reuse=None):
    """
    Applies a linear transformation to the incoming data.

    :param input_: 2D or 1D data (Tensor, ndarray)
    :param output_size: size of output matrix or vector
    :param stddev:
    :param is_range:
    :param squeeze:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope('Linear', reuse=reuse):
        if type(input_) == np.ndarray:
            shape = input_.shape
        else:
            shape = input_.get_shape().as_list()

        is_vector = False
        if len(shape) == 1:
            is_vector = True
            input_ = tf.reshape(input_, [1, -1])
            input_size = shape[0]
        elif len(shape) == 2:
            input_size = shape[1]
        else:
            raise ValueError('Linear expects shape[1] of input: %s' % str(shape))

        w_name = '%s_w' % name if name else None
        b_name = '%s_b' % name if name else None

        w = tf.get_variable(w_name, [input_size, output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        mul = tf.matmul(input_, w)

        if is_range:
            def identity_initializer(tensor):
                # noinspection PyUnusedLocal
                def _initializer(shape_, dtype=tf.float32, partition_info=None):
                    return tf.identity(tensor)

                return _initializer

            range_ = tf.range(output_size, 0, -1)
            b = tf.get_variable(b_name, [output_size], tf.float32,
                                identity_initializer(tf.cast(range_, tf.float32)))
        else:
            b = tf.get_variable(b_name, [output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))

        if squeeze:
            output = tf.squeeze(tf.nn.bias_add(mul, b))
        else:
            output = tf.nn.bias_add(mul, b)

        if is_vector:
            return tf.reshape(output, [-1])
        else:
            return output


def outer_product(*inputs):
    """

    :param inputs: a list of 1D Tensor (vector)
    :return:
    """
    inputs = list(inputs)
    order = len(inputs)
    for idx, input_ in enumerate(inputs):
        if len(input_.get_shape()) == 1:
            inputs[idx] = tf.reshape(input_, [-1, 1] if idx % 2 == 0 else [1, -1])

    output = None
    if order == 2:
        output = tf.multiply(inputs[0], inputs[1])
    elif order == 3:
        size = []
        for i in range(order):
            size.append(inputs[i].get_shape()[0])

        output = tf.zeros(size)
        mul = tf.multiply(inputs[0], inputs[1])
        for i in range(size[-1]):
            output = tf.scatter_add(output, [0, 0, i], mul)

    return output


# noinspection PyUnusedLocal
def scalar_mul(x, beta, name=None):
    return x * beta


# noinspection PyUnusedLocal
def scalar_div(x, beta, name=None):
    return x / beta


def smooth_cosine_similarity(m, v):
    """

    :param m: a 2D Tensor (matrix)
    :param v: a 1D Tensor (vector)
    :return:
    """
    shape_x = m.get_shape().as_list()
    shape_y = v.get_shape().as_list()
    if shape_x[1] != shape_y[0]:
        raise ValueError('Smooth cosine similarity is expecting same dimension')

    m_norm = tf.sqrt(tf.reduce_sum(tf.pow(m, 2), 1))
    v_norm = tf.sqrt(tf.reduce_sum(tf.pow(v, 2)))
    m_dot_v = tf.matmul(m, tf.reshape(v, [-1, 1]))
    similarity = tf.div(tf.reshape(m_dot_v, [-1]), m_norm * v_norm + 1e-3)

    return similarity
