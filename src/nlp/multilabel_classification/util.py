from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score


def print_evaluation_scores(y_true, y_pred):
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1 score (macro avg):', f1_score(y_true, y_pred, average='macro'))
    print('F1 score (micro avg):', f1_score(y_true, y_pred, average='micro'))
    print('F1 score (weighted avg):', f1_score(y_true, y_pred, average='weighted'))
    print('Precision (macro avg):', average_precision_score(y_true, y_pred, average='macro'))
    print('Precision (micro avg):', average_precision_score(y_true, y_pred, average='micro'))
    print('Precision (weighted avg):', average_precision_score(y_true, y_pred, average='weighted'))
    print('Recall (macro avg):', recall_score(y_true, y_pred, average='macro'))
    print('Recall (micro avg):', recall_score(y_true, y_pred, average='micro'))
    print('Recall (weighted avg):', recall_score(y_true, y_pred, average='weighted'))


def print_labels(binarizer, x, y_true, y_pred, n=3):
    y_true_inverse = binarizer.inverse_transform(y_true)
    y_pred_inverse = binarizer.inverse_transform(y_pred)
    for i in range(n):
        print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            x[i],
            ', '.join(y_true_inverse[i]),
            ', '.join(y_pred_inverse[i])
        ))


def print_words_for_tag(classifier, tag, tag_classes, index_to_words, n=5):
    """

    :param classifier: trained classifier
    :param tag: a tag
    :param tag_classes: a list of tag names from `MultiLabelBinarizer`
    :param index_to_words: map
    :param n: top number of results to print
    :return: nothing, just print top n positive and top n negative words for given tag
    """
    print('Tag:\t{}'.format(tag))

    # extract estimator from the classifier for the given tag
    estimator = classifier.estimators_[tag_classes.index(tag)]

    # extract feature coefficients from the estimator
    coefficients = estimator.coef_[0].argsort()

    top_positive_words = [index_to_words[idx] for idx in coefficients[-n:]]
    top_negative_words = [index_to_words[idx] for idx in coefficients[:n]]

    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}'.format(', '.join(top_negative_words)))
