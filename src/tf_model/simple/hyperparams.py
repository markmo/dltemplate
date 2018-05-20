def get_constants():
    return {
        'n_input': 28 * 28,  # number of input parameters
        'n_classes': 10,  # number of output classes
        'n_hidden': 50,
        'learning_rate': 0.001,
        'early_stop_threshold': 0.98,
        'n_epochs': 50,  # number of training iterations
        'batch_size': 128,
        'n_report_steps': 1  # number of steps between reporting
    }
