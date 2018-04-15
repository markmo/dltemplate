from homemade.classes import Dense, Dropout, ReLU


def network_builder(inp, constants):
    # Dropout only required if overfitting. Default `keep_prob` is 1.
    return [
        Dense(inp.shape[1], constants['n_hidden1'], initialization='relu'),
        ReLU(),
        Dropout(constants['keep_prob']),
        Dense(constants['n_hidden1'], constants['n_hidden2'], initialization='relu'),
        ReLU(),
        Dropout(constants['keep_prob']),
        Dense(constants['n_hidden2'], constants['n_classes'], initialization='relu')
    ]
