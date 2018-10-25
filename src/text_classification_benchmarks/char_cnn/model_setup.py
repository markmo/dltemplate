from common.util_keras import TQDMProgressCallback
from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard
from keras.layers import AlphaDropout, Concatenate, Convolution1D, Dense, Embedding, GlobalMaxPooling1D, Input
from keras.models import Model
import os


class CharCNN(object):
    """
    Class to implement the character-level CNN as described in
    Kim et al. 2015 (https://arxiv.org/abs/1508.06615)

    Their model has been adapted to perform text classification
    instead of language modelling by replacing subsequent recurrent
    layers with dense layer(s) to perform softmax over classes.
    """
    def __init__(self, input_size, alphabet_size, embedding_size, conv_layers,
                 fully_connected_layers, n_classes, keep_prob, model_filename,
                 optimizer='adam', loss='categorical_crossentropy'):
        """

        :param input_size: (int) size of input features
        :param alphabet_size: (int) size of alphabets to create embeddings for
        :param embedding_size: (int)
        :param conv_layers: (list[list[int]]) list of convolution layers
        :param fully_connected_layers: (list[list[int]]) list of fully connected layers
        :param n_classes: (int)
        :param keep_prob: (float) dropout keep probability
        :param model_filename: (str)
        :param optimizer: (str) optimizer function
        :param loss: (str) loss function
        """
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.n_classes = n_classes
        self.keep_prob = keep_prob
        self.model_filename = model_filename
        self.optimizer = optimizer
        self.loss = loss
        self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.input_size, ), name='sent_input', dtype='int64')
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)
        convolution_output = []
        for n_filters, filter_width in self.conv_layers:
            conv = Convolution1D(filters=n_filters, kernel_size=filter_width, activation='tanh',
                                 name='Conv1D_{}_{}'.format(n_filters, filter_width))(x)
            pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(n_filters, filter_width))(conv)
            convolution_output.append(pool)

        x = Concatenate()(convolution_output)
        for layer in self.fully_connected_layers:
            x = Dense(layer, activation='selu', kernel_initializer='lecun_normal')(x)
            x = AlphaDropout(self.keep_prob)(x)

        preds = Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=preds)
        if os.path.exists(self.model_filename):
            model.load_weights(self.model_filename)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, n_epochs, batch_size):  # , checkpoint_every=100):
        """

        :param x_train: (ndarray) training set inputs
        :param y_train: (ndarray) training set labels
        :param x_val: (ndarray) validation set inputs
        :param y_val: (ndarray) validation set labels
        :param n_epochs: (int) number training epochs
        :param batch_size: (int) batch size
        # :param checkpoint_every: (int) interval for logging to TensorBoard
        :return: None
        """
        # Create callbacks
        # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=checkpoint_every, batch_size=batch_size,
        #                           write_graph=False, write_grads=True, write_images=False,
        #                           embeddings_freq=checkpoint_every, embeddings_layer_names=None)
        print('Training...')
        checkpoint = ModelCheckpoint(self.model_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                       epochs=n_epochs, batch_size=batch_size, verbose=2,
                       callbacks=[TQDMProgressCallback(), checkpoint])
        #               callbacks=[tensorboard])

    def test(self, x_test, y_test, batch_size):
        """

        :param x_test: (ndarray) testing set inputs
        :param y_test: (ndarray) testing set labels
        :param batch_size: (int)
        :return: None
        """
        # Evaluate inputs
        self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    def predict(self, x_test, batch_size):
        """
        Generates output predictions for the input samples.

        :param x_test: (ndarray) testing set inputs
        :param batch_size: (int)
        :return: (ndarray) predictions
        """
        return self.model.predict(x_test, batch_size)
