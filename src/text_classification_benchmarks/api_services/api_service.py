from collections import deque
import numpy as np
import time


class ApiService(object):

    def __init__(self, classes, max_api_calls=None, verbose=False):
        self.max_api_calls = max_api_calls
        self.verbose = verbose
        self.elapsed = deque(maxlen=1000)
        if type(classes) is np.ndarray:
            self.classes = classes.tolist()
        else:
            self.classes = classes

    def mean_latency(self):
        return np.mean(self.elapsed) if self.elapsed else 0

    def predict(self, utterance):
        pass

    def predict_label(self, utterance):
        tic = time.time()
        intent = self.predict(utterance)
        toc = time.time()
        self.elapsed.append(toc - tic)
        try:
            return self.classes.index(intent) if intent else -1
        except Exception as e:
            print('ERR:', e)
            return -1

    def predict_batch(self, val_df):
        y_pred = []
        for i, utterance in enumerate(val_df.utterance.values):
            label = self.predict_label(utterance)
            y_pred.append(label)
            if self.verbose:
                print('{}, {}, {}'.format(utterance, self.classes[label], self.classes[val_df.label.values[i]]))
                print()

            if self.max_api_calls and i == self.max_api_calls - 1:  # save on API calls
                break

        return y_pred
