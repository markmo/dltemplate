from argparse import ArgumentParser
from collections import defaultdict
from common.load_data import load_tagged_sentences
from common.util import merge_dict
from keras_model.pos_tagger.hyperparams import get_constants
from keras_model.pos_tagger.model_setup import bidirectional_model_builder, model_builder
from keras_model.pos_tagger.util import compute_test_accuracy, draw, get_word_counts, to_matrix, train
from sklearn.model_selection import train_test_split


def run(constant_overwrites):
    constants = merge_dict(get_constants(), constant_overwrites)
    data, all_tags = load_tagged_sentences()

    draw(data[11])
    draw(data[10])
    draw(data[7])

    all_words, word_counts = get_word_counts(data)

    # let's measure what fraction of data words are in the dictionary
    coverage = float(sum(word_counts[w] for w in all_words)) / sum(word_counts.values())
    print('Coverage = %.5f' % coverage)

    # Build a mapping from tokens to integer ids. Our model operates on a word level,
    # processing one word per RNN step. This means we'll have to deal with far larger
    # vocabulary.
    # Luckily for us, we only receive those words as input, i.e. we don't have to
    # predict them. This means we can have a large vocabulary for free by using word
    # embeddings.
    word_to_id = defaultdict(lambda: 1, {word: i for i, word in enumerate(all_words)})
    tag_to_id = {tag: i for i, tag in enumerate(all_tags)}

    batch_words, batch_tags = zip(*[zip(*sentence) for sentence in data[-3:]])

    print('word ids:')
    print(to_matrix(batch_words, word_to_id))
    print('tag ids:')
    print(to_matrix(batch_tags, tag_to_id))

    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    if constants['bidirectional']:
        model = bidirectional_model_builder(all_words, all_tags, constants)
    else:
        model = model_builder(all_words, all_tags, constants)

    print('')
    print(model.summary())
    print('')

    train(model, train_data, test_data, all_tags, word_to_id, tag_to_id, constants)

    # Measure final accuracy on the whole test set.
    acc = compute_test_accuracy(model, test_data, word_to_id, tag_to_id)
    print('\n\nFinal accuracy: %.5f' % acc)

    assert acc > 0.94, 'Accuracy should be better than that'


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run Keras RNN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.add_argument('--bidirectional', dest='bidirectional', help='bidirectional flag', action='store_true')
    parser.set_defaults(retrain=False)
    parser.set_defaults(bidirectional=False)
    args = parser.parse_args()
    run(vars(args))
