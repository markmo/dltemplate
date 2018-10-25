from abc import ABCMeta
from collections import deque
import numpy as np
from scipy import sparse
from scipy.sparse import issparse
import six
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize, LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
import time


class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, alpha=1.0, c_penalty=1.0, max_iter=10000):
        self.alpha = alpha
        self.c_penalty = c_penalty
        self.max_iter = max_iter
        self.models = []
        self.classes = None
        self.class_count = None
        self.ratios = None
        self.elapsed = deque(maxlen=1000)

    def fit(self, x, y):
        x, y = check_X_y(x, y, 'csr')
        _, n_features = x.shape
        binarizer = LabelBinarizer()
        y = binarizer.fit_transform(y)
        self.classes = binarizer.classes_
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)

        # So we don't have to cast X to floating point
        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64
        y = y.astype(np.float64)

        # Count raw events from data
        n_effective_classes = y.shape[1]
        self.class_count = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios = np.full((n_effective_classes, n_features), self.alpha, dtype=np.float64)
        self._compute_ratios(x, y)

        for i in range(n_effective_classes):
            x_i = x.multiply(self.ratios[i])
            svm = LinearSVC(C=self.c_penalty, max_iter=self.max_iter)
            y_i = y[:, i]
            svm.fit(x_i, y_i)
            self.models.append(svm)

        return self

    def mean_latency(self):
        return np.mean(self.elapsed) if self.elapsed else 0

    def predict(self, x):
        tic = time.time()
        n_effective_classes = self.class_count.shape[0]
        n_examples = x.shape[0]
        d = np.zeros((n_effective_classes, n_examples))
        for i in range(n_effective_classes):
            x_i = x.multiply(self.ratios[i])
            d[i] = self.models[i].decision_function(x_i)

        label = self.classes[np.argmax(d, axis=0)]
        toc = time.time()
        self.elapsed.append(toc - tic)
        return label

    def _compute_ratios(self, x, y):
        """ Count feature occurrences and compute ratios. """
        if np.any((x.data if issparse(x) else x) < 0):
            raise ValueError('Input x must be non-negative')

        self.ratios += safe_sparse_dot(y.T, x)  # ratio + feature_occurrence_c
        normalize(self.ratios, norm='l1', axis=1, copy=False)
        # noinspection PyTypeChecker
        self.ratios = np.apply_along_axis(calc_row, axis=1, arr=self.ratios)
        check_array(self.ratios)
        self.ratios = sparse.csr_matrix(self.ratios)


def calc_row(r):
    return np.log(np.divide(r, 1 - r))
