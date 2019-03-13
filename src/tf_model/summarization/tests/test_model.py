import numpy as np
from tf_model.summarization.model_setup import build_models
from .test_util import generate_dataset, one_hot_decode, predict_sequence


def test_model():
    n_features = 51
    n_encoder_tokens = 6
    n_decoder_tokens = 3
    model, encoder_model, decoder_model = build_models(n_features, n_features, 128)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Generate training dataset
    x1, x2, y = generate_dataset(n_encoder_tokens, n_decoder_tokens, n_features, 100000)
    print(x1.shape, x2.shape, y.shape)

    # Train model
    model.fit([x1, x2], y, epochs=1)

    # Evaluate LSTM
    total, correct = 100, 0
    for _ in range(total):
        x1, x2, y = generate_dataset(n_encoder_tokens, n_decoder_tokens, n_features, 1)
        target = predict_sequence(x1, encoder_model, decoder_model, n_decoder_tokens, n_features)
        if np.array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
            correct += 1

    print('Accuracy: %.2f%%' % float(correct / total * 100))

    # Spot check some examples
    for _ in range(10):
        x1, x2, y = generate_dataset(n_encoder_tokens, n_decoder_tokens, n_features, 1)
        target = predict_sequence(x1, encoder_model, decoder_model, n_decoder_tokens, n_features)
        print('X=%s y=%s, y_hat=%s' % (one_hot_decode(x1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
