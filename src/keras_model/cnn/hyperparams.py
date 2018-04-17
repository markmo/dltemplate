def get_constants():
    return {
        'filters': (16, 32, 32, 64),
        'kernel_size': (3, 3),
        'keep_prob': (0.25, 0.25, 0.5),
        'n_hidden': (256, 10),
        'learning_rate': 5e-3,
        'n_epochs': 10,
        'batch_size': 32,
        'model_filename': 'cifar.{0:03d}.hdf5'
    }
