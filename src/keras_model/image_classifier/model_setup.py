import keras


def model_builder(n_classes, constants):
    """
    You cannot train such a huge architecture from scratch with such a small dataset.

    But using fine-tuning of last layers of pre-trained network you can get a pretty
    good classifier very quickly.

    :param n_classes:
    :param constants:
    :return:
    """
    img_size = constants['img_size']

    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(img_size, img_size, 3),
                                           weights='imagenet' if constants['use_imagenet'] else None)

    # add global pooling as in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)

    # add new dense layer for our labels
    new_output = keras.layers.Dense(n_classes, activation='softmax')(new_output)

    model = keras.engine.training.Model(model.inputs, new_output)

    return model
