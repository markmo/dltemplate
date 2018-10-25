import numpy as np


class Data(object):
    """ Class to handle loading and processing of raw datasets """
    def __init__(self, x, y, alphabet, input_size, n_classes):
        self.x = x
        self.y = y
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.length = input_size
        self.n_classes = n_classes
        self.data = None
        self.dict = {}
        for i, ch in enumerate(alphabet):
            self.dict[ch] = i + 1

    def load_data(self):
        self.data = np.array(list(zip(self.y, self.x)))

    def get_all_data(self):
        """

        :return: (ndarray) data transformed from raw to indexed form with associated one-hot label
        """
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.n_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indices(s))
            c = int(c) - 1
            classes.append(one_hot[c])

        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indices(self, s):
        """
        Convert a string to character indices based on character dictionary.
        :param s: (str) string to be converted
        :return: (ndarray) indices of characters in s
        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]

        return str2idx
