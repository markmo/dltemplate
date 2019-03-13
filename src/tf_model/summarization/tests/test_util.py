from keras.utils import to_categorical
import numpy as np
from random import randint


def generate_dataset(n_in, n_out, cardinality, n_samples):
    x1, x2, y = [], [], []
    for _ in range(n_samples):
        # Generate source sequence
        source = generate_sequence(n_in, cardinality)

        # Define padded target sequence
        target = source[:n_out]
        target.reverse()

        # Create padded input target sequence
        target_in = [0] + target[:-1]

        # Encode
        src_encoded = to_categorical([source], num_classes=cardinality)[0]
        tgt_encoded = to_categorical([target], num_classes=cardinality)[0]
        tgt2_encoded = to_categorical([target_in], num_classes=cardinality)[0]

        # Store
        x1.append(src_encoded)
        x2.append(tgt2_encoded)
        y.append(tgt_encoded)

    return np.array(x1), np.array(x2), np.array(y)


def generate_sequence(length, n_unique):
    return [randint(1, n_unique - 1) for _ in range(length)]


def one_hot_decode(encoded_seq):
    return [np.argmax(vec) for vec in encoded_seq]


def predict_sequence(input_seq, encoder_model, decoder_model, n_decoder_tokens, n_steps):
    # Encode the input as state vectors
    states = encoder_model.predict(input_seq)

    # Start of sequence input
    target_seq = np.zeros((1, 1, n_decoder_tokens))

    # Collect predictions
    output = []
    for _ in range(n_steps):
        # Predict next char
        y_hat, h, c = decoder_model.predict([target_seq] + states)

        # Store prediction
        output.append(y_hat[0, 0, :])

        # Update target sequence
        target_seq = y_hat

        # Update states
        states = [h, c]

    return np.array(output)
