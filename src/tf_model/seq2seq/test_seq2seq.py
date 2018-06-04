from tf_model.seq2seq.util import generate_equations, get_symbol_to_id_mappings, sentence_to_ids


def test_generate_equations():
    allowed_operators = ['+', '-']
    dataset_size = 10
    for (input_, output_) in generate_equations(allowed_operators, dataset_size, 0, 100):
        assert type(input_) is str and type(output_) is str, 'Both parts must be strings'
        assert eval(input_) == int(output_), \
            'The (equation: {!r}, solution: {!r}) pair is incorrect.'.format(input_, output_)


def test_sentence_to_ids():
    sentences = [('123+123', 7), ('123+123', 8), ('123+123', 10)]
    expected_output = [([5, 6, 7, 3, 5, 6, 1], 7),
                       ([5, 6, 7, 3, 5, 6, 7, 1], 8),
                       ([5, 6, 7, 3, 5, 6, 7, 1, 2, 2], 8)]

    word2id, _ = get_symbol_to_id_mappings()

    for (sentence, padded_len), (sentence_ids, expected_length) in zip(sentences, expected_output):
        output, length = sentence_to_ids(sentence, word2id, padded_len)
        assert output == sentence_ids, \
            "Conversion of '{}' for padded_len={} to {} is incorrect.".format(sentence, padded_len, output)

        assert length == expected_length, \
            "Conversion of '{}' for padded_len={} has incorrect actual length {}.".format(sentence, padded_len, length)
