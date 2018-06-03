from argparse import ArgumentParser
from common.load_data import load_stack_overflow_dataset
from common.nlp_util import clean_text, get_counts, map_words_to_index
from common.nlp_util import tfidf_features, top_n, to_sparse_matrix
from common.util import merge_dict, plot_roc_auc
from nlp.multilabel_classification.hyperparams import get_constants
from nlp.multilabel_classification.model_setup import train_classifier
from nlp.multilabel_classification.util import print_evaluation_scores, print_labels, print_words_for_tag
from sklearn.preprocessing import MultiLabelBinarizer


def run(constant_overwrites):
    # nltk.download('stopwords')
    constants = merge_dict(get_constants(), constant_overwrites)
    x_train, y_train, x_val, y_val, x_test = load_stack_overflow_dataset()
    x_train = [clean_text(text) for text in x_train]
    x_val = [clean_text(text) for text in x_val]
    x_test = [clean_text(text) for text in x_test]
    tag_counts = get_counts(y_train)
    word_counts = get_counts(x_train)
    dict_size = constants['dict_size']
    top_words = top_n(word_counts, dict_size)
    words_to_index = map_words_to_index(top_words)
    # index_to_words = map_index_to_words(top_words)
    # all_words = words_to_index.keys()

    x_train_bow = to_sparse_matrix(x_train, words_to_index, dict_size)
    x_val_bow = to_sparse_matrix(x_val, words_to_index, dict_size)
    # x_test_bow = to_sparse_matrix(x_test, words_to_index, dict_size)

    x_train_tfidf, x_val_tfidf, x_test_tfidf, tfidf_vocab = tfidf_features(x_train, x_val, x_test)
    tfidf_vocab_reverse = {i: word for word, i in tfidf_vocab.items()}

    classes = sorted(tag_counts.keys())
    n_classes = len(tag_counts)
    binarizer = MultiLabelBinarizer(classes=classes)
    y_train = binarizer.fit_transform(y_train)
    y_val = binarizer.fit_transform(y_val)

    classifier_bow = train_classifier(x_train_bow, y_train)
    classifier_tfidf = train_classifier(x_train_tfidf, y_train)

    y_val_predicted_labels_bow = classifier_bow.predict(x_val_bow)
    y_val_predicted_scores_bow = classifier_bow.decision_function(x_val_bow)

    print('Bag-of-words metrics:')
    print_labels(binarizer, x_val, y_val, y_val_predicted_labels_bow)
    print_evaluation_scores(y_val, y_val_predicted_labels_bow)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(x_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(x_val_tfidf)

    print('TF-IDF metrics:')
    print_labels(binarizer, x_val, y_val, y_val_predicted_labels_tfidf)
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    # Plot ROC AUC BOW
    plot_roc_auc(y_val, y_val_predicted_scores_bow, n_classes)

    # Plot ROC AUC TF-IDF
    plot_roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)

    # print words for tag 'c'
    print_words_for_tag(classifier_tfidf, 'c', binarizer.classes, tfidf_vocab_reverse, n=5)

    # print words for tag 'c++'
    print_words_for_tag(classifier_tfidf, 'c++', binarizer.classes, tfidf_vocab_reverse, n=5)

    # print words for tag 'linux'
    print_words_for_tag(classifier_tfidf, 'linux', binarizer.classes, tfidf_vocab_reverse, n=5)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Multi-label Classification')
    parser.add_argument('--size', dest='dict_size', type=int, help='size of dictionary')
    args = parser.parse_args()
    run(vars(args))
