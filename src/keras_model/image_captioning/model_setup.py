import os
import tensorflow as tf
from tensorflow.contrib import keras

K = keras.backend
L = keras.layers


def cnn_encoder_builder():
    """
    We take the last hidden layer of InceptionV3 as an image embedding

    https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html

    :return:
    """
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input
    model = keras.models.Model(model.inputs, L.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model


def decoder_builder(data, constants):
    """
    Since our problem is to generate image captions, the RNN text generator
    should be conditioned on the image. The idea is to use image features
    as an initial state for the RNN instead of zeros.

    Remember that you should transform the image feature vector to the RNN
    hidden state size using a fully-connected layer before passing it to the RNN.

    During training, we feed ground truth tokens into the LSTM to get
    predictions of next tokens.

    Notice that we don't need to feed the last token (END) as input.

    http://cs.stanford.edu/people/karpathy/

    We need to calculate token probability for each time step of every example!
    That's why we flatten the states and apply dense layers to calculate all
    token logits at once.

    :param data:
    :param constants:
    :return:
    """
    class Decoder(object):
        # [batch_size, img_embed_size] of CNN image features
        img_embeds = tf.placeholder('float32', [None, data['img_embed_size']], name='img_embeds')

        # [batch_size, time steps] of word ids
        sentences = tf.placeholder('int32', [None, None], name='sentences')

        # image embedding -> bottleneck to reduce number of parameters
        img_embed_to_bottleneck = L.Dense(constants['img_embed_bottleneck'],
                                          input_shape=(None, data['img_embed_size']),
                                          activation='elu',
                                          name='img_embed_to_bottleneck')

        # image embedding bottleneck -> LSTM initial state
        img_embed_bottleneck_to_h0 = L.Dense(constants['lstm_units'],
                                             input_shape=(None, constants['img_embed_bottleneck']),
                                             activation='elu',
                                             name='img_embed_bottleneck_to_h0')

        # word -> embedding
        word_embed = L.Embedding(data['vocab_size'], constants['word_embed_size'], name='word_embed')

        # LSTM Cell (from TensorFlow)
        lstm = tf.nn.rnn_cell.LSTMCell(constants['lstm_units'])

        # LSTM output -> logits bottleneck to reduce model complexity
        token_logits_bottleneck = L.Dense(constants['logit_bottleneck'],
                                          input_shape=(None, constants['lstm_units']),
                                          activation='elu',
                                          name='token_logits_bottleneck')

        # logits bottleneck -> logits for next token prediction
        token_logits = L.Dense(data['vocab_size'], input_shape=(None, constants['logit_bottleneck']),
                               name='token_logits')

        # Initial LSTM cell state of shape (None, LSTM_UNITS),
        # condition on `img_embeds` placeholder
        c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))

        # Embed all tokens but the last for LSTM input,
        # remember that Embedding is callable,
        # use `sentences` placeholder as input
        word_embeds = word_embed(sentences[:, :-1])

        # During training we use ground truth tokens (`word_embeds`) as context
        # for next token prediction. That means we know all the inputs for
        # our LSTM and can get all the hidden states with one TensorFlow
        # operation (`tf.nn.dynamic_run`).
        # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
        hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                             initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

        # Now we need to calculate token logits for all the hidden states

        # First, we reshape `hidden_states` to [-1, LSTM_UNITS]
        flat_hidden_states = tf.reshape(hidden_states, [-1, constants['lstm_units']])

        # Then, we calculate logits for next tokens using `token_logits_bottleneck`
        # and `token_logits` layers.
        flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))

        # Then, we flatten the ground truth token ids. Remember that we predict
        # next tokens for each time step. Use `sentences` placeholder.
        flat_ground_truth = tf.reshape(sentences[:, 1:], [-1])

        # We need to know where we have real tokens (not padding) in `flat_ground_truth`.
        # We don't want to propagate the loss for padded output tokens. Fill
        # `flat_loss_mask` with 1.0 for real tokens (not `pad_idx`) and 0.0 otherwise.
        flat_loss_mask = tf.cast(tf.not_equal(flat_ground_truth, data['pad_idx']), dtype=tf.float32)

        # Compute cross-entropy between `flat_ground_truth` and `flat_token_logits`
        # predicted by the LSTM.
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_ground_truth,
                                                              logits=flat_token_logits)

        # Compute average `xent` over tokens with nonzero `flat_loss_mask`.
        # We don't want to account misclassification of PAD tokens -
        # PAD tokens are for batching purposes only!
        loss = tf.reduce_sum(xent * flat_loss_mask) / tf.reduce_sum(flat_loss_mask)

    '''
    vocab = data['vocab']
    img_embed_size = data['img_embed_size']
    pad_idx = data['pad_idx']
    img_embed_bottleneck = constants['img_embed_bottleneck']
    lstm_units = constants['lstm_units']
    word_embed_size = constants['word_embed_size']
    logit_bottleneck = constants['logit_bottleneck']

    class decoder:

        # [batch_size, img_embed_size] of CNN image features
        img_embeds = tf.placeholder('float32', [None, img_embed_size])

        # [batch_size, time steps] of word ids
        sentences = tf.placeholder('int32', [None, None])

        # image embedding -> bottleneck to reduce the number of parameters
        img_embed_to_bottleneck = L.Dense(img_embed_bottleneck,
                                          input_shape=(None, img_embed_size),
                                          activation='elu')

        # image embedding bottleneck -> LSTM initial state
        img_embed_bottleneck_to_h0 = L.Dense(lstm_units,
                                             input_shape=(None, img_embed_bottleneck),
                                             activation='elu')

        # word -> embedding
        word_embed = L.Embedding(len(vocab), word_embed_size)

        # LSTM Cell (from TensorFlow)
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_units)

        # LSTM output -> logits bottleneck to reduce model complexity
        token_logits_bottleneck = L.Dense(logit_bottleneck,
                                          input_shape=(None, lstm_units),
                                          activation="elu")

        # logits bottleneck -> logits for next token prediction
        token_logits = L.Dense(len(vocab),
                               input_shape=(None, logit_bottleneck))

        # Initial LSTM cell state of shape (None, lstm_units),
        # condition on `img_embeds` placeholder
        c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))

        # Embed all tokens but the last for LSTM input,
        # remember that Embedding is callable,
        # use `sentences` placeholder as input
        word_embeds = word_embed(sentences[:, :-1])

        # During training we use ground truth tokens (`word_embeds`) as context
        # for next token prediction. That means we know all the inputs for
        # our LSTM and can get all the hidden states with one TensorFlow
        # operation (`tf.nn.dynamic_rnn`).
        # `hidden_states` has a shape of [batch_size, time steps, lstm_units]
        hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                             initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

        # Now we need to calculate token logits for all the hidden states.

        # First, we reshape `hidden_states` to [-1, lstm_units].
        flat_hidden_states = tf.reshape(hidden_states, [-1, lstm_units])

        # Then, we calculate logits for next tokens using `token_logits_bottleneck`
        # and `token_logits` layers.
        flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))

        # Then, we flatten the ground truth token ids. Remember that we predict
        # next tokens for each time step. Use `sentences` placeholder.
        flat_ground_truth = tf.reshape(sentences[:, 1:], [-1])

        # We need to know where we have real tokens (not padding) in `flat_ground_truth`.
        # We don't want to propagate the loss for padded output tokens. Fill
        # `flat_loss_mask` with 1.0 for real tokens (not `pad_idx`) and 0.0 otherwise.
        flat_loss_mask = tf.cast(tf.not_equal(flat_ground_truth, pad_idx), dtype=tf.float32)

        # Compute cross-entropy between `flat_ground_truth` and `flat_token_logits`
        # predicted by the LSTM.
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_ground_truth,
                                                              logits=flat_token_logits)

        # Compute average `xent` over tokens with nonzero `flat_loss_mask`.
        # We don't want to account misclassification of PAD tokens -
        # PAD tokens are for batching purposes only!
        loss = tf.reduce_sum(xent * flat_loss_mask) / tf.reduce_sum(flat_loss_mask)

    return decoder
    '''
    return Decoder


'''
class Decoder(object):
    """
    Since our problem is to generate image captions, the RNN text generator
    should be conditioned on the image. The idea is to use image features
    as an initial state for the RNN instead of zeros.

    Remember that you should transform the image feature vector to the RNN
    hidden state size using a fully-connected layer before passing it to the RNN.

    During training, we feed ground truth tokens into the LSTM to get
    predictions of next tokens.

    Notice that we don't need to feed the last token (END) as input.

    http://cs.stanford.edu/people/karpathy/

    We need to calculate token probability for each time step of every example!
    That's why we flatten the states and apply dense layers to calculate all
    token logits at once.
    """

    def __init__(self, data, constants):
        self.img_embed_size = data['img_embed_size']
        self.vocab_size = data['vocab_size']
        self.pad_idx = data['pad_idx']
        self.img_embed_bottleneck = constants['img_embed_bottleneck']
        self.lstm_units = constants['lstm_units']
        self.word_embed_size = constants['word_embed_size']
        self.logit_bottleneck = constants['logit_bottleneck']
        self._img_embeds = None
        self._sentences = None
        self._img_embed_to_bottleneck = None
        self._img_embed_bottleneck_to_h0 = None
        self._word_embed = None
        self._lstm = None
        self._token_logits_bottleneck = None
        self._token_logits = None
        self._c0 = None
        self._word_embeds = None
        self._hidden_states = None
        self._flat_hidden_states = None
        self._flat_token_logits = None
        self._flat_ground_truth = None
        self._flat_loss_mask = None
        self._loss = None

    @property
    def img_embeds(self):
        """
        [batch_size, img_embed_size] of CNN image features

        :return:
        """
        if self._img_embeds is None:
            self._img_embeds = tf.placeholder('float32', [None, self.img_embed_size], name='img_embeds')

        return self._img_embeds

    @property
    def sentences(self):
        """
        [batch_size, time steps] of word ids

        :return:
        """
        if self._sentences is None:
            self._sentences = tf.placeholder('int32', [None, None], name='sentences')

        return self._sentences

    @property
    def img_embed_to_bottleneck(self):
        """
        image embedding -> bottleneck to reduce number of parameters

        :return:
        """
        if self._img_embed_to_bottleneck is None:
            self._img_embed_to_bottleneck = L.Dense(self.img_embed_bottleneck,
                                                    input_shape=(None, self.img_embed_size),
                                                    activation='elu',
                                                    name='img_embed_to_bottleneck')

        return self._img_embed_to_bottleneck

    @property
    def img_embed_bottleneck_to_h0(self):
        """
        image embedding bottleneck -> LSTM initial state

        :return:
        """
        if self._img_embed_bottleneck_to_h0 is None:
            self._img_embed_bottleneck_to_h0 = L.Dense(self.lstm_units,
                                                       input_shape=(None, self.img_embed_bottleneck),
                                                       activation='elu',
                                                       name='img_embed_bottleneck_to_h0')

        return self._img_embed_bottleneck_to_h0

    @property
    def word_embed(self):
        """
        word -> embedding

        :return:
        """
        if self._word_embed is None:
            self._word_embed = L.Embedding(self.vocab_size, self.word_embed_size, name='word_embed')

        return self._word_embed

    @property
    def lstm(self):
        """
        LSTM Cell (from TensorFlow)

        :return:
        """
        if self._lstm is None:
            self._lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_units)

        return self._lstm

    @property
    def token_logits_bottleneck(self):
        """
        LSTM output -> logits bottleneck to reduce model complexity

        :return:
        """
        if self._token_logits_bottleneck is None:
            self._token_logits_bottleneck = L.Dense(self.logit_bottleneck,
                                                    input_shape=(None, self.lstm_units),
                                                    activation='elu',
                                                    name='token_logits_bottleneck')

        return self._token_logits_bottleneck

    @property
    def token_logits(self):
        """
        logits bottleneck -> logits for next token prediction

        :return:
        """
        if self._token_logits is None:
            self._token_logits = L.Dense(self.vocab_size, input_shape=(None, self.logit_bottleneck),
                                         name='token_logits')

        return self._token_logits

    @property
    def c0(self):
        """
        Initial LSTM cell state of shape (None, LSTM_UNITS),
        condition on `img_embeds` placeholder

        :return:
        """
        if self._c0 is None:
            self._c0 = self.img_embed_bottleneck_to_h0(self.img_embed_to_bottleneck(self.img_embeds))

        return self._c0

    @property
    def h0(self):
        if self._c0 is None:
            self._c0 = self.c0

        return self._c0

    @property
    def word_embeds(self):
        """
        Embed all tokens but the last for LSTM input,
        remember that Embedding is callable,
        use `sentences` placeholder as input

        :return:
        """
        if self._word_embeds is None:
            self._word_embeds = self.word_embed(self.sentences[:, :-1])

        return self._word_embeds

    @property
    def hidden_states(self):
        """
        During training we use ground truth tokens (`word_embeds`) as context
        for next token prediction. That means we know all the inputs for
        our LSTM and can get all the hidden states with one TensorFlow
        operation (`tf.nn.dynamic_run`).

        `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].

        :return:
        """
        if self._hidden_states is None:
            c0 = h0 = self.c0
            self._hidden_states, _ = tf.nn.dynamic_rnn(self.lstm, self.word_embeds,
                                                       initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

        return self._hidden_states

    # Now we need to calculate token logits for all the hidden states

    @property
    def flat_hidden_states(self):
        """
        First, we reshape `hidden_states` to [-1, LSTM_UNITS]

        :return:
        """
        if self._flat_hidden_states is None:
            self._flat_hidden_states = tf.reshape(self.hidden_states, [-1, self.lstm_units])

        return self._flat_hidden_states

    @property
    def flat_token_logits(self):
        """
        Then, we calculate logits for next tokens using `token_logits_bottleneck`
        and `token_logits` layers.

        :return:
        """
        if self._flat_token_logits is None:
            self._flat_token_logits = self.token_logits(self.token_logits_bottleneck(self.flat_hidden_states))

        return self._flat_token_logits

    @property
    def flat_ground_truth(self):
        """
        Then, we flatten the ground truth token ids. Remember that we predict
        next tokens for each time step. Use `sentences` placeholder.
        
        :return: 
        """
        if self._flat_ground_truth is None:
            self._flat_ground_truth = tf.reshape(self.sentences[:, 1:], [-1])

        return self._flat_ground_truth

    @property
    def flat_loss_mask(self):
        """
        We need to know where we have real tokens (not padding) in `flat_ground_truth`.
        We don't want to propagate the loss for padded output tokens. Fill
        `flat_loss_mask` with 1.0 for real tokens (not `pad_idx`) and 0.0 otherwise.
        
        :return: 
        """
        if self._flat_loss_mask is None:
            self._flat_loss_mask = tf.cast(tf.not_equal(self.flat_ground_truth, self.pad_idx), dtype=tf.float32)

        return self._flat_loss_mask

    @property
    def loss(self):
        if self._loss is None:
            # Compute cross-entropy between `flat_ground_truth` and `flat_token_logits`
            # predicted by the LSTM.
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.flat_ground_truth,
                                                                  logits=self.flat_token_logits)

            # Compute average `xent` over tokens with nonzero `flat_loss_mask`.
            # We don't want to account misclassification of PAD tokens -
            # PAD tokens are for batching purposes only!
            self._loss = tf.reduce_sum(xent * self.flat_loss_mask) / tf.reduce_sum(self.flat_loss_mask)

        return self._loss
'''


def model_builder(sess, constants, decoder, saver):
    """
    Construct a graph of the final model. It works as follows:

    * Take an image as an input and embed it
    * Condition the LSTM on that embedding
    * Predict the next token given a START input token
    * Use predicted token as an input to next time step
    * Iterate until an END token is predicted

    :param sess:
    :param constants:
    :param decoder:
    :param saver:
    :return:
    """
    class Model(object):
        img_size = constants['img_size']
        lstm_units = constants['lstm_units']

        # CNN encoder
        encoder, preprocess_for_model = cnn_encoder_builder()

        # Keras applications corrupt our graph, so we restore trained weights
        saver.restore(sess, os.path.abspath('weights'))

        # containers for current LSTM state
        lstm_c = tf.Variable(tf.zeros([1, lstm_units]), name='cell')
        lstm_h = tf.Variable(tf.zeros([1, lstm_units]), name='hidden')

        # input images
        input_images = tf.placeholder('float32', [1, img_size, img_size, 3], name='images')

        # get image embeddings
        img_embeds = encoder(input_images)

        # initialize LSTM state conditioned on image
        init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
        init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)

        # current word index
        current_word = tf.placeholder('int32', [1], name='current_input')

        # embedding for current word
        word_embed = decoder.word_embed(current_word)

        # apply LSTM cell, get new LSTM states
        new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

        # compute logits for next token
        new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))

        # compute probabilities for next token
        new_probs = tf.nn.softmax(new_logits)

        # `one_step` outputs probabilities of next token and updates LSTM hidden state
        one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)

    return Model
