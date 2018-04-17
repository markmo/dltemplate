from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, Dropout
from keras.layers import Flatten, Input, InputLayer, MaxPooling2D, Reshape
from keras.models import Model, Sequential


def network_builder(constants):
    img_shape = constants['img_shape']
    enc_filters = constants['enc_filters']
    dec_filters = constants['dec_filters']
    kernel_size = constants['kernel_size']
    keep_prob = constants['keep_prob']
    code_size = constants['code_size']

    # encoder
    enc = Sequential()
    enc.add(InputLayer(img_shape))
    enc.add(Conv2D(filters=enc_filters[0], kernel_size=kernel_size, padding='same'))
    enc.add(Activation('elu'))
    enc.add(Conv2D(filters=enc_filters[1], kernel_size=kernel_size, padding='same'))
    enc.add(Activation('elu'))
    enc.add(MaxPooling2D(pool_size=(2, 2)))
    enc.add(Dropout(keep_prob))
    enc.add(Conv2D(filters=enc_filters[2], kernel_size=kernel_size, padding='same'))
    enc.add(Activation('elu'))
    enc.add(Conv2D(filters=enc_filters[3], kernel_size=kernel_size, padding='same'))
    enc.add(Activation('elu'))
    enc.add(MaxPooling2D(pool_size=(2, 2)))
    enc.add(Flatten())
    enc.add(Dense(code_size))

    # decoder
    dec = Sequential()
    dec.add(InputLayer((code_size,)))
    dec.add(Dense(2*2*enc_filters[3]))
    dec.add(Reshape((2, 2, enc_filters[3])))
    dec.add(Conv2DTranspose(filters=dec_filters[0], kernel_size=kernel_size, strides=2,
                            activation='elu', padding='same'))
    dec.add(Conv2DTranspose(filters=dec_filters[1], kernel_size=kernel_size, strides=2,
                            activation='elu', padding='same'))
    dec.add(Conv2DTranspose(filters=dec_filters[2], kernel_size=kernel_size, strides=2,
                            activation='elu', padding='same'))
    dec.add(Conv2DTranspose(filters=dec_filters[3], kernel_size=kernel_size, strides=2,
                            activation='elu', padding='same'))

    return enc, dec


def model_builder(network, constants):
    encoder, decoder = network(constants)
    inp = Input(constants['img_shape'])
    code = encoder(inp)
    reconstruction = decoder(code)
    autoencoder = Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    return autoencoder, encoder, decoder
