from keras.layers import Bidirectional, Dense, Embedding, InputLayer, SimpleRNN, TimeDistributed
from keras.models import Sequential


def model_builder(all_words, all_tags, constants):
    """
    We'll use the high-level Keras interface to define our recurrent neural network.
    It is as simple as you can get with RNN, albeit somewhat constraining for complex
    tasks like seq2seq.

    By default, all Keras RNNs apply to a whole sequence of inputs and produce a
    sequence of hidden states (`return_sequences=True` or just the last hidden state
    (`return_sequences=False`). All the recurrence is happening under the hood.

    At the top of our model we need to apply a Dense layer to each time-step independently.
    By default `keras.layers.Dense` would apply once to all time-steps concatenated.
    We use `keras.layers.TimeDistributed` to modify the Dense layer so that it applies
    across both batch and time axes.

    :param all_words:
    :param all_tags:
    :param constants:
    :return:
    """
    model = Sequential()
    model.add(InputLayer([None], input_dtype='int32'))
    model.add(Embedding(len(all_words), constants['embedding_size']))
    model.add(SimpleRNN(constants['rnn_units'], return_sequences=True))

    # add top layer that predicts tag probabilities
    stepwise_dense = Dense(len(all_tags), activation='softmax')
    stepwise_dense = TimeDistributed(stepwise_dense)
    model.add(stepwise_dense)

    return model


def bidirectional_model_builder(all_words, all_tags, constants):
    """
    Since we're analyzing a full sequence, it's legal for us to look into future data.

    A simple way to achieve that is to go both directions at once, making a bidirectional RNN.

    In Keras you can achieve that both manually (using two LSTMs and Concatenate) and by using
    `keras.layers.Bidirectional`.

    This one works just as `TimeDistributed` we saw before: you wrap it around a recurrent layer
    (SimpleRNN now and LSTM or GRU later), and it actually creates two layers under the hood.

    :param all_words:
    :param all_tags:
    :param constants:
    :return:
    """
    model = Sequential()
    model.add(InputLayer([None], input_dtype='int32'))
    model.add(Embedding(len(all_words), constants['embedding_size']))
    model.add(Bidirectional(SimpleRNN(constants['rnn_units'], return_sequences=True)))

    # add top layer that predicts tag probabilities
    stepwise_dense = Dense(len(all_tags), activation='softmax')
    stepwise_dense = TimeDistributed(stepwise_dense)
    model.add(stepwise_dense)

    return model
