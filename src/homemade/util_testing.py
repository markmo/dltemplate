import numpy as np


# noinspection SpellCheckingInspection
def eval_numerical_gradient(f, x, verbose=False, h=0.00001):
    """
    Evaluates gradient df/dx via finite differences:

    df/dx ~ (f(x+h) - f(x-h)) / 2h

    .. math::

    \boldsymbol{d}^\top\! \nabla f(\boldsymbol{x}) \approx \frac{1}{2 \varepsilon}(f(\boldsymbol{x} + \varepsilon \cdot \boldsymbol{d}) - f(\boldsymbol{x} - \varepsilon \cdot \boldsymbol{d}))

    :param f:
    :param x:
    :param verbose:
    :param h:
    :return:
    """
    # fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)

    # iterate over all indices in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])

        it.iternext()  # step to next dimension

    return grad
