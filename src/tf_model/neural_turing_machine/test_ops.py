import numpy as np
import tensorflow as tf
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.platform import googletest
from tf_model.neural_turing_machine.ops import circular_convolution, smooth_cosine_similarity


class CircularConvolutionTest(TensorFlowTestCase):

    def test_circular_convolution(self):
        v = constant([1, 2, 3, 4, 5, 6, 7], dtype=tf.float32)
        k = constant([0, 0, 1], dtype=tf.float32)
        for use_gpu in [True, False]:
            with self.test_session(use_gpu=use_gpu):
                loss = circular_convolution(v, k).eval()
                self.assertAllEqual(loss, [7, 1, 2, 3, 4, 5, 6])


class SmoothCosineSimilarityTest(TensorFlowTestCase):

    def test_smooth_cosine_similarity(self):
        m = constant([[1, 2, 3], [2, 2, 2], [3, 2, 1], [0, 2, 4]], dtype=np.float32)
        v = constant([2, 2, 2], dtype=np.float32)
        for use_gpu in [True, False]:
            with self.test_session(use_gpu=use_gpu):
                loss = smooth_cosine_similarity(m, v).eval()
                self.assertAllClose(loss, [0.92574867671153,
                                           0.99991667361053,
                                           0.92574867671153,
                                           0.77454667246876])


if __name__ == '__main__':
    googletest.main()
