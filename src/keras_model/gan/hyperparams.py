def get_constants():
    return {
        'n_hidden_units': 64,
        'keep_prob': 0.4,
        'z_dim': 100,
        'momentum': 0.9,
        'discriminator_learning_rate': 0.0008,
        'discriminator_decay': 6e-8,
        'adversarial_learning_rate': 0.0004,
        'adversarial_decay': 3e-8,
        'n_epochs': 2000,
        'batch_size': 128,
        'n_report_steps': 500
    }
