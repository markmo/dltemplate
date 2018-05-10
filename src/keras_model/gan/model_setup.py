from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import RMSprop


def discriminator_builder(img_w, img_h, constants):
    n_hidden_units = constants['n_hidden_units']
    keep_prob = constants['keep_prob']

    print('n_hidden_units:', n_hidden_units)
    print('keep_prob:', keep_prob)

    # define inputs
    inputs = Input((img_w, img_h, 1))

    # convolutional layers
    conv1 = Conv2D(n_hidden_units * 1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(keep_prob)(conv1)

    conv2 = Conv2D(n_hidden_units * 2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(keep_prob)(conv2)

    conv3 = Conv2D(n_hidden_units * 4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(keep_prob)(conv3)

    conv4 = Conv2D(n_hidden_units * 8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(keep_prob)(conv4))

    # output layer
    output = Dense(1, activation='sigmoid')(conv4)

    # model definition
    model = Model(inputs=inputs, outputs=output)

    return model


def get_discriminator(img_w, img_h, constants):
    discriminator = discriminator_builder(img_w, img_h, constants)

    print('discriminator_learning_rate:', constants['discriminator_learning_rate'])
    print('discriminator_decay:', constants['discriminator_decay'])

    optimizer = RMSprop(lr=constants['discriminator_learning_rate'],
                        decay=constants['discriminator_decay'],
                        clipvalue=1.0)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return discriminator


def generator_builder(constants):
    z_dim = constants['z_dim']
    n_hidden_units = constants['n_hidden_units']
    keep_prob = constants['keep_prob']
    momentum = constants['momentum']

    print('z_dim:', z_dim)
    print('n_hidden_units:', n_hidden_units)
    print('keep_prob:', keep_prob)
    print('momentum:', momentum)

    # define inputs
    inputs = Input((z_dim,))

    # first dense layer
    dense1 = Dense(7 * 7 * 64)(inputs)
    dense1 = BatchNormalization(momentum=momentum, axis=-1)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7, 7, 64))(dense1)
    dense1 = Dropout(keep_prob)(dense1)

    # deconvolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(n_hidden_units / 2), kernel_size=5, padding='same', activation=None)(conv1)
    conv1 = BatchNormalization(momentum=momentum, axis=-1)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(n_hidden_units / 4), kernel_size=5, padding='same', activation=None)(conv2)
    conv2 = BatchNormalization(momentum=momentum, axis=-1)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = Conv2DTranspose(int(n_hidden_units / 8), kernel_size=5, padding='same', activation=None)(conv2)
    conv3 = BatchNormalization(momentum=momentum, axis=-1)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # output layer
    output = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)

    # model definition
    model = Model(inputs=inputs, outputs=output)

    return model


def adversarial_builder(generator, discriminator, constants):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    print('adversarial_learning_rate:', constants['adversarial_learning_rate'])
    print('adversarial_decay:', constants['adversarial_decay'])

    optimizer = RMSprop(lr=constants['adversarial_learning_rate'],
                        decay=constants['adversarial_decay'],
                        clipvalue=1.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
