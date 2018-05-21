import numpy as np


def get_constants():
    return {
        'n_hidden': 200,
        'embedding_dim': 200,
        'batch_size': 32,
        'n_epochs': 4,
        'learning_rate': 0.005,
        'learning_rate_decay': np.sqrt(2),
        'dropout_keep_prob': 0.5
    }
