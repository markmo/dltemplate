from keras_model.wake_word_detection.util import insert_ones, is_overlapping
import numpy as np


def test_is_overlapping():
    assert not is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
    assert is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])


def test_ones():
    np.random.seed(5)
    ty = 1375  # number of time steps in the output of our model
    arr1 = insert_ones(np.zeros((1, ty)), 9700)
    insert_ones(arr1, 4251)
    assert [arr1[0][1333], arr1[0][634], arr1[0][635]] == [0, 1, 0]
