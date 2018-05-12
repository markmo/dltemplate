import numpy as np


def zero_pad(x, pad):
    """
    Pad all images of the dataset x with zeros. The padding is applied to
    the height and width of an image.

    :param x: numpy array of shape (m, n_H, n_W, n_C) representing a batch
              of m images
    :param pad: integer, amount of padding around each image on vertical
                and horizontal dimensions
    :return: padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    return np.pad(x, ((0,), (pad,), (pad,), (0,)), 'constant', constant_values=0)


def conv_single_step(a_slice_prev, w, b):
    """
    Apply one filter defined by parameter w on a single slice (`a_slice_prev`)
    of the output activation of the previous layer.

    :param a_slice_prev: slice of input data of shape (f, f, n_C_prev), f is filters
    :param w: Weight parameters contained in a window of shape (f, f, n_C_prev)
    :param b: Bias parameters contained in a window of shape (1, 1, 1)
    :return: scalar, result of convolving the sliding window (w, b) on a
             slice x of the input data
    """
    s = np.multiply(a_slice_prev, w) + b
    z = np.sum(s)
    return z


def conv_forward(a_prev, wx, b, constants):
    """
    Implements the forward pass for a convolution function.

    :param a_prev: output activations of the previous layer,
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param wx: weights, numpy array of shape (f, f, n_C_prev, n_C), `f` is filters
    :param b: biases, numpy array of shape (1, 1, 1, n_C)
    :param constants: dict of hyperparameters
    :return: z - conv output, numpy array of shape (m, n_H, n_W, n_C)
             cache - cache of values needed for the `conv_backward` function
    """
    (m, n_h_prev, n_w_prev, n_c_prev) = a_prev.shape
    (f, f, n_c_prev, n_c) = wx.shape
    stride = constants['stride']
    pad = constants['pad']

    n_h = int((n_h_prev - f + 2 * pad) / stride) + 1
    n_w = int((n_w_prev - f + 2 * pad) / stride) + 1
    z = np.zeros((m, n_h, n_w, n_c))
    a_prev_pad = zero_pad(a_prev, pad)

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # find corners of the current slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # use the corners to define the (3D) slice of `a_prev_pad`
                    a_slice_prev = a_prev_pad[i][vert_start:vert_end, horiz_start:horiz_end, :]

                    # convolve the (3D) slice with the correct filter w and bias b,
                    # to get back one output neuron
                    z[i, h, w, c] = conv_single_step(a_slice_prev, wx[..., c], b[..., c])

    assert(z.shape == (m, n_h, n_w, n_c))
    cache = (a_prev, wx, b, constants)

    return z, cache


def pool_forward(a_prev, constants, mode='max'):
    """
    Implement the forward pass of the pooling layer

    The pooling (POOL) layer reduces the height and width of the input. It helps
    reduce computation, as well as helps make feature detectors more invariant
    to its position in the input. The two types of pooling layers are:

    * Max-pooling layer: slides an (f, f) window over the input and stores
      the max value of the window in the output.
    * Average-pooling layer: slides an (f, f) window over the input and stores
      the average value of the window in the output.

    These pooling layers have no parameters for backpropagation to train. However,
    they have hyperparameters such as the window size f. This specifies the height
    and width of the f x f window you would compute a max or average over.

    :param a_prev: input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param constants: dict of hyperparameters
    :param mode: pooling mode, defined as a string ['max', 'average']
    :return: a - output of pooling layer, numpy array of shape (m, n_H, n_W, n_C)
             cache - cache used in the backward pass of the pooling layer,
                     contains the input and `constants`
    """
    (m, n_h_prev, n_w_prev, n_c_prev) = a_prev.shape
    f = constants['filters']
    stride = constants['stride']

    # define the dimensions of the output
    n_h = int(1 + (n_h_prev - f) / stride)
    n_w = int(1 + (n_w_prev - f) / stride)
    n_c = n_c_prev

    # initialize output matrix a
    a = np.zeros((m, n_h, n_w, n_c))

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # find the corners of the current slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # use the corners to define the current slice on
                    # the ith training example of a_prev, channel c
                    a_prev_slice = a_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # compute the pooling operation on the slice
                    if mode == 'max':
                        a[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        a[i, h, w, c] = np.mean(a_prev_slice)

    # store the input and constants in `cache` for `pool_backward`
    cache = (a_prev, constants)
    assert(a.shape == (m, n_h, n_w, n_c))

    return a, cache


def conv_backward(dz, cache):
    """
    Implement the backward propagation for a convolution function

    :param dz: gradient of the cost with respect to the output of
               the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    :param cache: cache of values needed for the `conv_backward`,
                  output of `conv_forward`
    :return: da_prev - gradient of the cost with respect to the input
                       of the conv layer (A_prev), numpy array of shape
                       (m, n_H_prev, n_W_prev, n_C_prev)
             dw - gradient of the cost with respect to the weights of
                  the conv layer (W), numpy array of shape (f, f, n_C_prev, n_C)
             db - gradient of the cost with respect to the biases of
                  the conv layer (b), numpy array of shape (1, 1, 1, n_C)
    """
    (a_prev, wx, b, constants) = cache
    (m, n_h_prev, n_w_prev, n_c_prev) = a_prev.shape
    (f, f, n_c_prev, n_c) = wx.shape
    pad = constants['pad']
    (m, n_h, n_w, n_c) = dz.shape

    # initialize da_prev, dw, db with the correct shapes
    da_prev = np.zeros((m, n_h_prev, n_w_prev, n_c_prev))
    dw = np.zeros((f, f, n_c_prev, n_c))
    db = np.zeros((1, 1, 1, n_c))

    # pad a_prev and da_prev
    a_prev_pad = zero_pad(a_prev, pad)
    da_prev_pad = zero_pad(da_prev, pad)

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # find the corners of the current slice
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # use the corners to define the slice from `a_prev_pad`
                    a_slice = a_prev_pad[i][vert_start:vert_end, horiz_start:horiz_end, :]

                    # update gradients for the window and the filter's parameters
                    da_prev_pad[i][vert_start:vert_end, horiz_start:horiz_end, :] += wx[:, :, :, c] * dz[i, h, w, c]
                    # Where $W_c$ is a filter and $dZ_{hw}$ is a scalar corresponding to the gradient
                    # of the cost with respect to the output of the conv layer Z at the hth row and
                    # wth column (corresponding to the dot product taken at the ith stride left and
                    # jth stride down). Note that at each time, we multiply the same filter $W_c$ by
                    # a different dZ when updating dA. We do so mainly because when computing the
                    # forward propagation, each filter is dotted and summed by a different a_slice.
                    # Therefore when computing the backprop for dA, we are just adding the gradients
                    # of all the a_slices.

                    dw[:, :, :, c] += a_slice * dz[i, h, w, c]
                    # Where $a_{slice}$ corresponds to the slice which was used to generate the
                    # activation $Z_{ij}$. Hence, this ends up giving us the gradient for $W$ with
                    # respect to that slice. Since it is the same $W$, we will just add up all such
                    # gradients to get $dW$.

                    db[:, :, :, c] += dz[i, h, w, c]
                    # db is computed by summing $dZ$. In this case, you are just summing over all the
                    # gradients of the conv output (Z) with respect to the cost.

        # set the ith training example's `da_prev` to the unpadded `da_prev_pad`
        da_prev[i, :, :, :] = da_prev_pad[i][pad:-pad, pad:-pad, :]

    assert(da_prev.shape == (m, n_h_prev, n_w_prev, n_c_prev))

    return da_prev, dw, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    This function creates a "mask" matrix which keeps track of where the
    maximum of the matrix is. True (1) indicates the position of the maximum
    in X, the other entries are False (0).

    We keep track of the position of the max because this is the input value
    that ultimately influenced the output, and therefore the cost. Backprop
    is computing gradients with respect to the cost, so anything that influences
    the ultimate cost should have a non-zero gradient. So, backprop will
    "propagate" the gradient back to this particular input value that had
    influenced the cost.

    :param x: array of shape (f, f)
    :return: mask - array of the same shape as window, contains a True at
                    the position corresponding to the max entry of x
    """
    return x == np.max(x)


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape.

    In max pooling, for each input window, all the "influence" on the output
    came from a single input value--the max. In average pooling, every element
    of the input window has equal influence on the output.

    :param dz: input scalar
    :param shape: the shape (n_H, n_W) of the output matrix for which we want
                  to distribute the value of dz
    :return: array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_h, n_w) = shape

    # compute the value to distribute on the matrix
    average = dz / (n_h * n_w)

    # create a matrix where every entry is the average value
    return np.ones(shape) * average


def pool_backward(da, cache, mode='max'):
    """
    Implements the backward pass of the pooling layer.

    Even though a pooling layer has no parameters for backprop to update,
    you still need to back-propagate the gradient through the pooling layer
    in order to compute gradients for layers that came before the pooling layer.

    :param da: gradient of cost with respect to the output of the pooling layer,
               same shape as a
    :param cache: cache output from the forward pass of the pooling layer,
                  contains the layer's input and hyperparameters
    :param mode: pooling mode, defined as a string ['max', 'average']
    :return: da_prev - gradient of cost with respect to the input of the pooling
                       layer, same shape as a_prev
    """
    (a_prev, constants) = cache
    f = constants['filters']

    # retrieve dimensions from `da`'s shape
    m, n_h, n_w, n_c = da.shape

    da_prev = np.zeros(a_prev.shape)

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # find the corners of the current slice
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    if mode == 'max':
                        # use the corners and `c` to define the current slice from `a_prev`
                        a_prev_slice = a_prev[i][vert_start:vert_end, horiz_start:horiz_end, c]

                        # create the mask from `a_prev_slice`
                        mask = create_mask_from_window(a_prev_slice)

                        # set da_prev to be da_prev + (the mask multiplied by the correct entry of da)
                        da_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, da[i, h, w, c])

                    elif mode == 'average':
                        da_ = da[i, h, w, c]

                        # define the shape of the filter as `f`x`f`
                        shape = (f, f)

                        # distribute it to get the correct slice of `da_prev`,
                        # i.e. add the distributed value of `da`
                        da_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da_, shape)

    assert(da_prev.shape == a_prev.shape)

    return da_prev
