from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from nltk.corpus import stopwords
import numpy as np
from pathlib import Path
import pickle
import re

ROOT_DIR = Path(__file__).parent


def clean_text(text, contractions, remove_stopwords=True):
    text = text.lower()
    new_text = []
    for word in text.split():
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)

    text = ' '.join(new_text)
    text = re.sub(r'https?://.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    if remove_stopwords:
        stopword_set = set(stopwords.words('english'))
        new_text = [w for w in text.split() if w not in stopword_set]
        text = ' '.join(new_text)

    return text


def decode_sequence(input_seq, encoder_model, decoder_model, n_decoder_tokens,
                    target_token_index, reverse_target_char_index, max_decoder_seq_length):
    # Encode the input as state vectors
    states = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, n_decoder_tokens))

    # Populate the first character of target sequence with the "start character"
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # To simplify, here we assume a batch of 1
    stop_condition = False
    decoded_sent = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sent += sampled_char

        # Exit condition: reach max length or find "stop character"
        if sampled_char == '\n' or len(decoded_sent) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, n_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states = [h, c]

    return decoded_sent


def get_texts_and_chars(stories):
    input_texts = []
    target_texts = []
    input_chars = set()
    target_chars = set()
    for story in stories:
        input_text = story['story']
        target_text = story['highlights']

        # We use "\t" as the "start sequence" character for
        # the targets and "\n" as the "end sequence" character
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_chars:
                input_chars.add(char)

        for char in target_text:
            if char not in target_chars:
                target_chars.add(char)

    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))
    return input_texts, target_texts, input_chars, target_chars


def get_indices(input_texts, target_texts, input_chars, target_chars, n_encoder_tokens, n_decoder_tokens,
                max_enc_seq_length, max_dec_seq_length):
    input_token_index = {char: i for i, char in enumerate(input_chars)}
    target_token_index = {char: i for i, char in enumerate(target_chars)}
    encoder_input_data = np.zeros((len(input_texts), max_enc_seq_length, n_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_dec_seq_length, n_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_dec_seq_length, n_decoder_tokens), dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for j, char in enumerate(input_text):
            encoder_input_data[i, j, input_token_index[char]] = 1.

        for j, char in enumerate(target_text):
            # `decoder_target_data` is ahead of `decoder_input_data` by one timestep
            decoder_input_data[i, j, target_token_index[char]] = 1.
            if j > 0:
                # `decoder_target_data` will be ahead by one timestep
                # and will not include the "start character"
                decoder_target_data[i, j - 1, target_token_index[char]] = 1.

    reverse_input_char_index = {i: char for char, i in input_token_index.items()}
    reverse_target_char_index = {i: char for char, i in target_token_index.items()}
    return (input_token_index, target_token_index, reverse_input_char_index, reverse_target_char_index,
            encoder_input_data, decoder_input_data, decoder_target_data)


def prepare_data(stories):
    input_texts, target_texts, input_chars, target_chars = get_texts_and_chars(stories)
    n_encoder_tokens = len(input_chars)
    n_decoder_tokens = len(target_chars)
    max_enc_seq_length = max([len(t) for t in input_texts])
    max_dec_seq_length = max([len(t) for t in target_texts])

    print('Number samples:', len(input_texts))
    print('Number unique input tokens:', n_encoder_tokens)
    print('Number unique output tokens:', n_decoder_tokens)
    print('Max sequence length for inputs:', max_enc_seq_length)
    print('Max sequence length for outputs:', max_dec_seq_length)

    (input_token_index, target_token_index, reverse_input_char_index, reverse_target_char_index,
     encoder_input_data, decoder_input_data, decoder_target_data) = \
        get_indices(input_texts, target_texts, input_chars, target_chars, n_encoder_tokens, n_decoder_tokens,
                    max_enc_seq_length, max_dec_seq_length)

    with open(ROOT_DIR / 'data' / 'reverse_input_char_index.pkl', 'wb') as f:
        pickle.dump(reverse_input_char_index, f)

    with open(ROOT_DIR / 'data' / 'reverse_target_char_index.pkl', 'wb') as f:
        pickle.dump(reverse_target_char_index, f)

    with open(ROOT_DIR / 'data' / 'target_token_index.pkl', 'wb') as f:
        pickle.dump(target_token_index, f)

    with open(ROOT_DIR / 'data' / 'lengths.csv', 'w') as f:
        f.write('{},{},{},{}'.format(n_encoder_tokens, n_decoder_tokens, max_enc_seq_length, max_dec_seq_length))

    return (encoder_input_data, decoder_input_data, decoder_target_data,
            n_encoder_tokens, n_decoder_tokens, target_token_index,
            reverse_input_char_index, reverse_target_char_index,
            max_dec_seq_length, input_texts)


def train(model, encoder_input_data, decoder_input_data, decoder_target_data,
          n_epochs, batch_size, model_dir):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    checkpoint = ModelCheckpoint(str(model_dir / 'model-{epoch:03d}-{acc:.3f}-{val_acc:.3f}.h5'),
                                 save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size, epochs=n_epochs, validation_split=0.2,
              callbacks=[early_stop, checkpoint, reduce_lr])
