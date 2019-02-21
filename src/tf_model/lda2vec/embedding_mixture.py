import numpy as np
import tensorflow as tf


class EmbeddingMixture(object):

    def __init__(self, n_docs, n_topics, n_dim, temperature=1.0, w_in=None, factors_in=None, name=''):
        self.n_docs = n_docs
        self.temperature = temperature
        self.name = name
        scalar = 1 / np.sqrt(n_docs + n_topics)
        if not isinstance(w_in, np.ndarray):
            self.doc_embedding = tf.Variable(tf.random_normal([n_docs, n_topics], mean=0, stddev=50 * scalar),
                                             name=self.name + '_doc_embedding')
        else:
            init = tf.constant(w_in)
            self.doc_embedding = tf.get_variable(self.name + '_doc_embedding', initializer=init)

        with tf.name_scope(self.name + '_topics'):
            if not isinstance(factors_in, np.ndarray):
                self.topic_embedding = tf.get_variable(self.name + '_topic_embedding', shape=[n_topics, n_dim],
                                                       dtype=tf.float32,
                                                       initializer=tf.orthogonal_initializer(gain=scalar))
            else:
                init = tf.constant(factors_in)
                self.topic_embedding = tf.get_variable(self.name + '_topic_embedding', initializer=init)

    def __call__(self, doc_ids=None, update_only_docs=False, softmax=True, *args, **kwargs):
        proportions = self.proportions(doc_ids, softmax=softmax)
        w_sum = tf.matmul(proportions, self.topic_embedding, name=self.name + '_docs_mul_topics')
        return w_sum

    def proportions(self, doc_ids=None, softmax=False):
        if doc_ids is None:
            w = self.doc_embedding
        else:
            w = tf.nn.embedding_lookup(self.doc_embedding, doc_ids, name=self.name + '_doc_proportions')

        if softmax:
            return tf.nn.softmax(w / self.temperature)
        else:
            return w
