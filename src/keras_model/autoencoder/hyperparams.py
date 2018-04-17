def get_constants():
    return {
        'code_size': 32,
        'enc_filters': (32, 64, 128, 256),
        'dec_filters': (128, 64, 32, 3),
        'kernel_size': (3, 3),
        'keep_prob': 0.25,
        'n_epochs': 25,
        'model_filename': 'autoencoder.{0:03d}.hdf5',
        'encoder_filename': 'encoder.h5',
        'decoder_filename': 'decoder.h5',
        'model': 'default'
    }
