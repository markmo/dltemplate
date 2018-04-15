import numpy as np
from tqdm import trange


# Loss functions

def softmax_crossentropy_with_logits(logits, reference_answers):
    """
    Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    """
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """
    Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    """
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (-ones_for_answers + softmax) / logits.shape[0]


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices = None
    if shuffle:
        indices = np.random.permutation(len(inputs))

    for start_idx in trange(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:(start_idx + batch_size)]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        yield inputs[excerpt], targets[excerpt]


def forward(network, inp):
    """
    Compute activations of all network layers by applying them sequentially.

    Return a list of activations for each layer.

    Make sure last activation corresponds to network logits.
    """
    activations = []
    for layer in network:
        inp = layer.forward(inp)
        activations.append(inp)

    assert len(activations) == len(network)
    return activations


def predict(network, inp):
    """
    Compute network predictions.
    """
    logits = forward(network, inp)[-1]
    return logits.argmax(axis=-1)


def train(network, inp, y):
    """
    Train your network on a given batch of X and y.

    You first need to run forward to get all layer activations. Then
    you can run layer.backward going from last to first layer.

    After you called backward for all layers, all Dense layers have
    already made one gradient step.
    """
    # Get the layer activations
    layer_activations = forward(network, inp)
    layer_inputs = [inp] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    # Propagate gradients through the network
    for i, layer in reversed(list(enumerate(network))):
        loss_grad = layer.backward(layer_inputs[i], loss_grad)

    return np.mean(loss)
