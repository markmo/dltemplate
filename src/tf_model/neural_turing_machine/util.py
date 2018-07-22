import numpy as np
import pprint
import sys
import tensorflow as tf


eps = 1e-12
pp = pprint.PrettyPrinter()


def argmax(x):
    index = 0
    max_num = x[index]
    for idx in range(1, len(x) - 1):
        if x[idx] > max_num:
            index = idx
            max_num = x[idx]

    return index, max_num


def gather(m_or_v, idx):
    if len(m_or_v.get_shape()) > 1:
        return tf.gather(m_or_v, idx)
    else:
        assert idx == 0, 'Error: idx should be 0 but %d' % idx
        return m_or_v


# noinspection PyBroadException
def matmul(x, y):
    """
    Compute matrix multiplication.

    :param x: a 2D Tensor (matrix)
    :param y: a 2D Tensor (matrix) or 1D Tensor (vector)
    :return:
    """
    try:
        return tf.matmul(x, y)
    except Exception:
        return tf.reshape(tf.matmul(x, tf.reshape(y, [-1, 1])), [-1])


def pprint(seq):
    seq = np.array(seq)
    seq = np.char.mod('%d', np.around(seq))
    seq[seq == '1'] = '#'
    seq[seq == '0'] = ' '
    print('\n'.join([''.join(x) for x in seq.tolist()]))


def progress(prog):
    bar_length = 20  # length of the progress bar
    status = ''
    if isinstance(prog, int):
        prog = float(prog)

    if not isinstance(prog, float):
        prog = 0
        status = 'error: progress var must be float\r\n'

    if prog < 0:
        prog = 0
        status = 'Halt...\r\n'
    elif prog >= 1:
        prog = 1
        status = 'Finished.\r\n'

    block = int(round(bar_length * prog))
    text = '\rPercent: [%s] %.2f%% %s' % ('#' * block + ' ' * (bar_length - block), prog * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


# noinspection PyBroadException
def softmax(x):
    """

    :param x: a 2D Tensor (matrix) or 1D Tensor (vector)
    :return:
    """
    try:
        return tf.nn.softmax(x + eps)
    except Exception:
        return tf.reshape(tf.nn.softmax(tf.reshape(x + eps, [1, -1])), [-1])
