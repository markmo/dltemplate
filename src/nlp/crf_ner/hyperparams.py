def get_constants():
    return {
        'algorithm': 'lbfgs',  # Gradient descent using the L-BFGS method
        'c1': 0.12,  # The coefficient for L1 regularization
        'c2': 0.01,  # The coefficient for L2 regularization
        'max_iterations': 100,  # The maximum number of iterations for the optimization algorithm
        'all_possible_transitions': True,  # When True, CRF suite generates transition features
                                           # that associate all of possible label pairs
    }
