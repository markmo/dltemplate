from argparse import ArgumentParser
from common.load_data import load_twitter_entities_dataset
from common.util import merge_dict
from nlp.crf_ner.hyperparams import get_constants
from nlp.crf_ner.util import build_sentences, evaluate, sentence_features, sentence_labels
from sklearn_crfsuite import CRF


def run(constant_overwrites):
    tokens_train, tags_train, tokens_val, tags_val, tokens_test, tags_test = load_twitter_entities_dataset()
    sents_train = build_sentences(tokens_train, tags_train)
    sents_val = build_sentences(tokens_val, tags_val)
    sents_test = build_sentences(tokens_test, tags_test)
    x_train = [sentence_features(sent) for sent in sents_train]
    y_train = [sentence_labels(sent) for sent in sents_train]
    x_val = [sentence_features(sent) for sent in sents_val]
    y_val = [sentence_labels(sent) for sent in sents_val]
    x_test = [sentence_features(sent) for sent in sents_test]
    y_test = [sentence_labels(sent) for sent in sents_test]
    constants = merge_dict(get_constants(), constant_overwrites)
    algorithm = constants['algorithm']
    c1 = constants['c1']
    c2 = constants['c2']
    max_iterations = constants['max_iterations']
    all_possible_transitions = constants['all_possible_transitions']

    print('')
    print('Hyperparameters')
    print('---------------')
    print('algorithm:', algorithm)
    print('c1:', c1)
    print('c2:', c2)
    print('max_iterations:', max_iterations)
    print('all_possible_transitions:', all_possible_transitions)

    # Build the CRF Model
    model = CRF(algorithm=algorithm,
                c1=c1,
                c2=c2,
                max_iterations=max_iterations,
                all_possible_transitions=all_possible_transitions)

    model.fit(x_train, y_train)

    print('-' * 20 + ' Train set quality: ' + '-' * 20)
    evaluate(model, x_train, y_train, short_report=False)

    print('-' * 20 + ' Validation set quality: ' + '-' * 20)
    evaluate(model, x_val, y_val, short_report=False)

    print('-' * 20 + ' Test set quality: ' + '-' * 20)
    evaluate(model, x_test, y_test, short_report=False)


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run CRF NER model')
    parser.add_argument('--algorithm', dest='algorithm', help='training algorithm')
    parser.add_argument('--c1', dest='c1', type=float, help='coefficient for L1 regularization')
    parser.add_argument('--c2', dest='c2', type=float, help='coefficient for L2 regularization')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        help='maximum number of iterations for optimization')
    parser.add_argument('--all-possible-transitions', dest='all_possible_transitions',
                        help='generate all label pairs', action='store_true')
    parser.set_defaults(all_possible_transitions=True)
    args = parser.parse_args()

    run(vars(args))
