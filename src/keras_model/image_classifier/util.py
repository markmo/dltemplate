from common.util import batch_generator, prepare_raw_bytes_for_model, raw_generator_with_label_from_tar
from common.util_keras import ModelSaveCallback, TQDMProgressCallback
import keras
import numpy as np


def compile_model(model, constants):
    # set all layers trainable by default
    for layer in model.layers:
        layer.trainable = True

    # fix deep layers (fine-tuning only last 50)
    for layer in model.layers[:-50]:
        layer.trainable = False

    # compile new model
    # we can take big lr here because we fixed first layers
    model.compile(
        loss='categorical_crossentropy',  # we train 102-way classification
        optimizer=keras.optimizers.adamax(lr=constants['learning_rate']),
        metrics=['accuracy']  # report accuracy during training
    )


def train_generator(tar_filename, files, labels, n_classes, constants):
    while True:  # so that Keras can loop through this as long as it wants
        items = raw_generator_with_label_from_tar(tar_filename, files, labels)
        for batch in batch_generator(items, constants['batch_size']):
            # prepare batch images
            batch_imgs = []
            batch_targets = []
            for raw, label in batch:
                img = prepare_raw_bytes_for_model(raw, constants['img_size'])
                batch_imgs.append(img)
                batch_targets.append(label)

            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)

            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = keras.utils.np_utils.to_categorical(batch_targets, n_classes)

            yield batch_imgs, batch_targets


def train(model, data, constants):
    """
    Training takes 2 hours on CPU. You're aiming for ~0.93 validation accuracy.

    :param data:
    :param model:
    :param constants:
    :return:
    """
    batch_size = constants['batch_size']
    train_files = data['train_files']
    train_labels = data['train_labels']
    test_files = data['test_files']
    test_labels = data['test_labels']
    n_classes = data['n_classes']
    tar_filename = constants['tar_filename']

    # fine tune for 2 epochs (full passes through all training data)
    # we make 2*8 epochs, where each epoch is 1/8 of our training data
    # to see progress more often
    model.fit_generator(
        train_generator(tar_filename, train_files, train_labels, n_classes, constants),
        steps_per_epoch=len(train_files) // batch_size // 8,
        epochs=constants['n_epochs'],
        validation_data=train_generator(tar_filename, test_files, test_labels, n_classes, constants),
        validation_steps=len(test_files) // batch_size // 4,
        callbacks=[TQDMProgressCallback(),
                   ModelSaveCallback(constants['model_filename'])],
        verbose=0,
        initial_epoch=constants['last_finished_epoch'] or 0
    )
