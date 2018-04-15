def get_constants():
    return {
        'img_shape': (28, 28, 1),
        'n_classes': 10,  # number of output classes
        'filters': (32, 64),
        'kernel_sizes': ((5, 5), (3, 3)),
        'n_hidden': 1024,
        'keep_prob': 0.5,
        'learning_rate': 0.001,
        'early_stop_threshold': 0.98,
        'epsilon': 1e-3,
        'n_epochs': 1000,  # number of training iterations
        'batch_size': 128,
        'n_report_steps': 1  # number of steps between reporting
    }
