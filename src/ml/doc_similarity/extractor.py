import logging
from ml.doc_similarity import string_util
import numpy as np


class Extractor(object):

    def __init__(self, config):
        """

        :param config: (dict) config dict
        """
        self.feature_name = self.__class__.__name__
        self.data_feature_fp = None
        self.config = config

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def extract_row(self, row):
        assert False, 'Please override function `extract_row`'

    def extract(self, df):
        logging.debug('Extracting values for feature: ' + self.feature_name)
        # values = []
        # for row in df.itertuples():
        #     feature_value = self.extract_row(row)
        #     values.append(feature_value)

        # using `apply` is faster than `iterrows` or `itertuples`
        # `axis=1` required to iterate by row; default by column
        # `tolist` required otherwise tries to assign multiple columns below
        values = df.apply(self.extract_row, axis=1).values.tolist()

        # convert to numpy array to get shape
        va = np.asarray(values)
        if len(va.shape) > 1 and va.shape[1] == 1:
            va = va.reshape(va.shape[0])
            values = va.tolist()

        feature_key = string_util.underscore_format(self.feature_name)
        logging.info('Creating feature: ' + feature_key)
        # df[feature_key] = np.asarray(values)
        df[feature_key] = values
        return df
