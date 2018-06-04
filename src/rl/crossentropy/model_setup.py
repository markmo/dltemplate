from sklearn.neural_network import MLPClassifier


def agent_builder(constants):
    n_hidden = constants('n_hidden')
    return MLPClassifier(hidden_layer_sizes=(n_hidden, n_hidden),
                         activation='tanh',
                         warm_start=True,  # keep progress between .fit(...) calls
                         max_iter=1  # make only 1 iteration on each .fit(...)
                         )
