from collections import deque
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import time


class LogisticRegressionModel(object):

    def __init__(self, x_train, y_train):
        self.count_vect = CountVectorizer()
        x_train_counts = self.count_vect.fit_transform(x_train)
        tfidf_transformer = TfidfTransformer()
        x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        self.classifier: LogisticRegression = model.fit(x_train_tfidf, y_train)
        self.elapsed = deque(maxlen=1000)

    def mean_latency(self):
        return np.mean(self.elapsed) if self.elapsed else 0

    def predict(self, utterance):
        tic = time.time()
        label = self.classifier.predict(self.count_vect.transform([utterance]))[0]
        toc = time.time()
        self.elapsed.append(toc - tic)
        return label
