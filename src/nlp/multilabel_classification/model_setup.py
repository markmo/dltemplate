from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def train_classifier(x_train, y_train):
    return OneVsRestClassifier(LogisticRegression(penalty='l1')).fit(x_train, y_train)
