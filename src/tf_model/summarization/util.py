from nltk.corpus import stopwords
import numpy as np
import re


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


def prepare_data(stories):
    input_texts = []
    target_texts = []
    input_chars = set()
    target_chars = set()
    for story in stories:
        input_text = story['story']
        target_text = ''
        for highlight in story['highlights']:
            target_text = highlight

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
    n_encoder_tokens = len(input_chars)
    n_decoder_tokens = len(target_chars)
    max_enc_seq_length = max([len(t) for t in input_texts])
    max_dec_seq_length = max([len(t) for t in target_texts])

    print('Number samples:', len(input_texts))
    print('Number unique input tokens:', n_encoder_tokens)
    print('Number unique output tokens:', n_decoder_tokens)
    print('Max sequence length for inputs:', max_enc_seq_length)
    print('Max sequence length for outputs:', max_dec_seq_length)

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

    return encoder_input_data, decoder_input_data, decoder_target_data, n_encoder_tokens, n_decoder_tokens


def train(model, encoder_input_data, decoder_input_data, decoder_target_data,
          n_epochs, batch_size, model_dir):
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size, epochs=n_epochs, validation_split=0.2)
    model.save(str(model_dir / 'model.h5'))
