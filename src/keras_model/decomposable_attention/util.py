import keras
import numpy as np


class BatchGenerator(keras.utils.Sequence):

    def __init__(self, x_set1, x_set2, y_set, batch_size):
        self.x1, self.x2, self.y = x_set1, x_set2, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))

    def __getitem__(self, idx):
        x1_batch = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        x2_batch = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [np.array(x1_batch), np.array(x2_batch)], np.array(y_batch)
