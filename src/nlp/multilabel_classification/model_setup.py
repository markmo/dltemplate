from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def train_classifier(x_train, y_train):
    lr_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2')
    return OneVsRestClassifier(lr_model).fit(x_train, y_train)
