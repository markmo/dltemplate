def get_constants():
    return {
        'img_size': 250,
        'use_imagenet': True,
        'learning_rate': 1e-2,  # we can use big lr because first layers of InceptionV3 model are fixed
        'n_epochs': 2 * 8,
        'last_finished_epoch': None,
        'batch_size': 32,
        'model_filename': 'flowers.{0:03d}.hdf5',
        'tar_filename': '102flowers.tgz'
    }
