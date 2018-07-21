from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from keras.layers import GRU, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


def build_model(input_shape):
    """

    :param input_shape: shape of the model's input data (using Keras conventions)
    :return: Keras model instance
    """
    x_input = Input(shape=input_shape)

    x = Conv1D(196, kernel_size=15, strides=4)(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.8)(x)

    # First GRU layer
    x = GRU(units=128, return_sequences=True)(x)
    x = Dropout(0.8)(x)
    x = BatchNormalization()(x)

    # Second GRU layer
    x = GRU(units=128, return_sequences=True)(x)
    x = Dropout(0.8)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)

    # Time-distributed dense layer
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs=x_input, outputs=x)

    return model


def fit(x, y, model, epochs=1, batch_size=5, learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01):
    optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(x, y, batch_size=batch_size, epochs=epochs)
