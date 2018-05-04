def get_constants():
    return {
        'img_size': 299,
        'img_embed_bottleneck': 120,
        'word_embed_size': 100,
        'lstm_units': 300,
        'logit_bottleneck': 120,
        'learning_rate': 0.001,
        'batch_size': 64,
        'n_epochs': 12,
        'n_batches_per_epoch': 1000,
        'n_validation_batches': 100,  # number of batches used for validation after each epoch
        'max_len': 20  # truncate long captions to speed up training
    }
