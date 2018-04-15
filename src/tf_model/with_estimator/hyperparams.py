def get_constants():
    return {
        'n_input': 28 * 28,  # number of input parameters
        'n_classes': 10,  # number of output classes
        'n_hidden1': 512,
        'n_hidden2': 256,
        'learning_rate': 0.001,
        'early_stop_threshold': 0.98,
        'epsilon': 1e-3,
        'n_epochs': 1000,  # number of training iterations
        'batch_size': 128,
        'n_report_steps': 1  # number of steps between reporting
    }
