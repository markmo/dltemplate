def get_constants():
    return {
        'rnn_units': 64,
        'embedding_size': 50,
        'batch_size': 32,
        'n_epochs': 5,
        'n_report_steps': 100,
        'model_filename': 'pos_tags.{0:03d}.hdf5'
    }
