import numpy as np


class Layer(object):
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:        output = layer.forward(input)
    - Propagate gradients through itself: grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """
    def __init__(self):
        """
        Here you can initialize layer parameters (if any) and auxiliary stuff.
        """
        # A dummy layer does nothing
        pass

    def forward(self, inp):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return inp

    def backward(self, inp, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.

        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):

        d loss / d x  = (d loss / d layer) * (d layer / d x)

        Luckily, you already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.

        If your layer has parameters (e.g. dense layer), you also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output,
        # but we'll write it more explicitly
        n_units = inp.shape[1]
        d_layer_d_input = np.eye(n_units)

        return np.dot(grad_output, d_layer_d_input)  # chain rule


class Dropout(Layer):

    def __init__(self, keep_prob):
        """
        Randomly shuts down some neurons in each iteration.
        """
        super().__init__()
        self.keep_prob = keep_prob
        self.d = None

    def forward(self, inp):
        """
        The outputs/activations of the previous layer are multiplied elementwise
        with a binary mask where the probability of each bit being 1 is based on
        `keep_prob`.
        """
        d = np.random.rand(inp.shape[0], inp.shape[1])
        d = d < self.keep_prob
        self.d = d
        return np.multiply(inp, d) / self.keep_prob

    def backward(self, inp, grad_output):
        """
        Multiply the derivative with the same dropout mask used during forward
        propagation, applying the same scaling.
        """
        return np.multiply(self.d, grad_output) / self.keep_prob


class ReLU(Layer):

    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        """
        super().__init__()
        pass

    def forward(self, inp):
        """
        Apply elementwise ReLU to [batch, input_units] matrix
        """
        return np.maximum(0, inp)

    def backward(self, inp, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        """
        relu_grad = inp > 0
        return grad_output * relu_grad


class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):
        """
        Leaky ReLU layer simply applies elementwise rectified linear unit to all inputs
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, inp):
        """
        Apply elementwise Leaky ReLU to [batch, input_units] matrix
        """
        return np.maximum(0, inp) - self.alpha * np.maximum(0, -inp)

    def backward(self, inp, grad_output):
        """
        Compute gradient of loss w.r.t. Leaky ReLU input
        """
        dx = np.ones_like(inp)
        dx[inp < 0] = self.alpha
        return grad_output * dx


# noinspection SpellCheckingInspection
class Dense(Layer):

    def __init__(self, input_units, output_units, learning_rate=0.1, initialization='default'):
        """
        A dense layer is a layer which performs a learned affine transformation:

        f(x) = <W*x> + b
        """
        super().__init__()
        self.learning_rate = learning_rate

        # Xavier initialisation helps ensure that weights that start too small
        # don't shrink until too small to be useful, and conversely, weights
        # that start too big don't explode until too big to be useful.
        # (Glorot et al.)
        if initialization == 'xavier':
            self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / (input_units + output_units))
        elif initialization == 'relu':
            # Initialization for ReLU neurons (He et al.)
            self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        else:
            self.weights = np.random.randn(input_units, output_units) * 0.01

        self.biases = np.zeros(output_units)

    def forward(self, inp):
        """
        Perform an affine transformation:

        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return np.dot(inp, self.weights) + self.biases

    def backward(self, inp, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(inp.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
