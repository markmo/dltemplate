import numpy as np
from scipy.stats import pearsonr

MISSING_VALUE_NUMERIC = -1


def count_one_bits(x):
    """
    Calculate the number of bits that are 1

    :param x: number
    :return: number of bits in `x`
    """
    n = 0
    while x:
        n += 1 if (x & 0x01) else 0
        x >>= 1

    return n


def int2binarystr(x):
    """
    Convert the number from decimal to binary

    :param x: (float) decimal number
    :return: (str) binary format of `x` as str
    """
    s = ''
    while x:
        s += '1' if (x & 0x01) else '0'
        x >>= 1

    return s[::-1]


def try_divide(x, y, val=0.):
    """Try to divide two numbers, safely accounting for div by zero."""
    if y != 0:
        val = float(x) / y

    return val


def corr(x, y_train):
    """
    Calculate the correlation between the specified feature and labels
    :param x: (ndarray) specified feature
    :param y_train: (ndarray) labels
    :return: (float) value of correlation
    """
    if dim(x) == 1:
        corr_ = pearsonr(x.flatten(), y_train)[0]
        if str(corr_) == 'nan':
            corr_ = 0.
    else:
        corr_ = 1.

    return corr_


def dim(x):
    if len(x.shape) == 1:
        return 1
    else:
        return x.shape[1]


def aggregate(data, modes):
    valid_modes = ['size', 'mean', 'std', 'max', 'min', 'median']
    if isinstance(modes, str):
        assert modes.lower() in valid_modes, 'Wrong aggregation mode: %s' % modes
        modes = [modes.lower()]
    elif isinstance(modes, list):
        for m in modes:
            assert m.lower() in valid_modes, 'Wrong aggregation mode: %s' % m
            modes = [m.lower() for m in modes]

    aggregators = [getattr(np, m) for m in modes]
    agg_values = []
    for agg in aggregators:
        try:
            s = agg(data)
        except ValueError:
            s = MISSING_VALUE_NUMERIC

        agg_values.append(s)

    return agg_values


def cut_prob(p):
    p[p > 1.0 - 1e-15] = 1.0 - 1e-15
    p[p < 1e-15] = 1e-15
    return p


def logit(p):
    assert isinstance(p, np.ndarray), 'type error'
    p = cut_prob(p)
    return np.log(p / (1. - p))


def logistic(y):
    assert isinstance(y, np.ndarray), 'type error'
    return np.exp(y) / (1. + np.exp(y))
