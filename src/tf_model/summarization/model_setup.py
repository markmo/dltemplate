from keras.layers import Dense, Input, LSTM
from keras.models import Model


def build_models(n_input, n_output, n_hidden):
    # define encoder
    enc_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_hidden, return_state=True)
    enc_outputs, state_h, state_c = encoder(enc_inputs)
    enc_states = [state_h, state_c]

    # define decoder
    dec_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_hidden, return_sequences=True, return_state=True)
    dec_outputs, _, _ = decoder_lstm(dec_inputs, initial_state=enc_states)
    decoder_dense = Dense(n_output, activation='softmax')
    dec_outputs = decoder_dense(dec_outputs)
    model = Model([enc_inputs, dec_inputs], dec_outputs)

    # define inference encoder
    encoder_model = Model(enc_inputs, enc_states)

    # define inference decoder
    dec_state_input_h = Input(shape=(n_hidden,))
    dec_state_input_c = Input(shape=(n_hidden,))
    dec_state_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = decoder_lstm(dec_inputs, initial_state=dec_state_inputs)
    dec_states = [state_h, state_c]
    dec_outputs = decoder_dense(dec_outputs)
    decoder_model = Model([dec_inputs] + dec_state_inputs, [dec_outputs] + dec_states)

    return model, encoder_model, decoder_model
