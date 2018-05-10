from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, LeakyReLU, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import adamax


def network_builder(constants):
    filters = constants['filters']
    kernel_size = constants['kernel_size']
    img_shape = constants['img_shape']
    keep_prob = constants['keep_prob']
    n_hidden = constants['n_hidden']

    model = Sequential()
    model.add(Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(0.1))

    model.add(Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(keep_prob[0]))

    model.add(Conv2D(filters=filters[2], kernel_size=kernel_size, padding='same'))
    model.add(LeakyReLU(0.1))

    model.add(Conv2D(filters=filters[3], kernel_size=kernel_size, padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(keep_prob[1]))
    model.add(Flatten())

    model.add(Dense(n_hidden[0]))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(keep_prob[2]))

    model.add(Dense(n_hidden[1]))
    model.add(Activation('softmax'))

    return model


def model_builder(network, constants):
    model = network(constants)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adamax(lr=constants['learning_rate']),
                  metrics=['accuracy'])
