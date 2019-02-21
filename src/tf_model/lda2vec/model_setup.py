from datetime import datetime
import numpy as np
from tf_model.lda2vec.embedding_mixture import EmbeddingMixture
from tf_model.lda2vec.util import dirichlet_likelihood
from tf_model.lda2vec.word_embedding import WordEmbedding

import tensorflow as tf


class Lda2vecModel(object):

    RESTORE_KEY = 'to_restore'

    def __init__(self, n_unique_docs, vocab_size, n_topics, freqs=None, load_embeds=False,
                 pretrained_embeddings=False, save_graph_def=True, embed_size=128, n_sampled=40,
                 learning_rate=0.001, lmbda=200.0, alpha=None, power=0.75, batch_size=500, log_dir='logs',
                 restore=False, w_in=None, factors_in=None, addtl_features_info=None, addtl_features_names=None):
        """

        :param n_unique_docs: (int) Number of unique documents in the dataset
        :param vocab_size: (int) Number of unique words/tokens in the dataset
        :param n_topics: (int) The set number of topics to cluster your data into
        :param freqs: (list, optional) list of length vocab_size with frequencies of each token
        :param load_embeds: (bool, optional) If true, will load embeddings from pretrained_embeddings variable
        :param pretrained_embeddings: (ndarray, optional) pretrained embeddings,
            shape should be (vocab_size, embed_size)
        :param save_graph_def: (bool, optional) If true, we will save the graph to `log_dir`
        :param embed_size: (int, optional) Dimension of the embeddings. This will be shared
            between docs, words, and topics.
        :param n_sampled: (int, optional) Negative sampling number for NCE Loss.
        :param learning_rate: (float, optional) Learning rate for the optimizer.
        :param lmbda: (float, optional) Strength of dirichlet prior
        :param alpha: (float, optional) alpha of dirichlet process (defaults to 1/n_topics)
        :param power: (float, optional) unigram sampler distortion
        :param batch_size: (int, optional)
        :param log_dir: (str, optional) Location for models to be saved. Note: datetime will be appended on each run.
        :param restore: (bool, optional) When True, will restore the model from the `log_dir` parameter's location.
        :param w_in: (ndarray, optional) Pretrained doc embedding weights,
            shape should be [n_unique_documents, embed_size]
        :param factors_in: (ndarray, optional) Pretrained topic embeddings, shape should be [n_topics, embed_size]
        :param addtl_features_info: (list, optional) list of the number of unique elements
            relating to the feature passed
        :param addtl_features_names: (list, optional) list of strings of the same length as `addtl_features_info`
            with names for the corresponding additional features
        """
        if addtl_features_info is None:
            addtl_features_info = []

        if addtl_features_names is None:
            addtl_features_names = []

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.moving_avgs = tf.train.ExponentialMovingAverage(0.9)
        self.n_unique_docs = n_unique_docs
        self.addtl_features_info = addtl_features_info
        self.n_addtl_features = len(self.addtl_features_info)
        self.addtl_features_names = addtl_features_names
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.freqs = freqs
        self.load_embeds = load_embeds
        self.pretrained_embeddings = pretrained_embeddings
        self.save_graph_def = save_graph_def
        self.log_dir = log_dir
        self.embed_size = embed_size
        self.n_sampled = n_sampled
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.alpha = alpha
        self.power = power
        self.batch_size = batch_size
        self.w_in = w_in
        self.factors_in = factors_in
        self.compute_normed = False
        self.n_batches = 0
        self.normed_embed_dict = {}
        self.idxs_in = None
        self.batch_array = None
        self.cosine_similarity = None
        if not restore:
            self.date = datetime.now().strftime('%y%m%d_%H%M')
            self.log_dir = '{}_{}'.format(self.log_dir, self.date)
            self.w_embed = WordEmbedding(self.embed_size, self.vocab_size, self.n_sampled,
                                         load_embeds=self.load_embeds,
                                         pretrained_embeddings=self.pretrained_embeddings,
                                         freqs=self.freqs,
                                         power=self.power)
            self.mixture_doc = EmbeddingMixture(self.n_unique_docs, self.n_topics, self.embed_size, name='mixture_doc')
            self.addtl_features_list = []
            for feature_i in range(self.n_addtl_features):
                self.addtl_features_list.append(
                    EmbeddingMixture(self.addtl_features_info[feature_i], self.n_topics, self.embed_size,
                                     name=self.addtl_features_names[feature_i]))

            handles = self._build_graph()
            for handle in handles:
                tf.add_to_collection(Lda2vecModel.RESTORE_KEY, handle)

            (self.x, self.y, self.docs, self.addtl_features, self.step, self.switch_loss, self.pivot,
             self.doc, self.context, self.loss_word2vec, self.fraction, self.loss_lda, self.loss,
             self.loss_avgs_op, self.optimizer, self.doc_embedding, self.topic_embedding, self.word_embedding,
             self.nce_weights, self.nce_biases, self.merged, *kg) = handles
            if len(kg) > 0:
                self.addtl_features_list = kg[:len(kg) // 2]
                self.feature_lookup = kg[len(kg) // 2:]
        else:
            meta_graph = log_dir + '/model.ckpt'
            tf.train.import_meta_graph(meta_graph + '.meta').restore(self.sess, meta_graph)
            handles = self.sess.graph.get_collection(Lda2vecModel.RESTORE_KEY)
            (self.x, self.y, self.docs, self.addtl_features, self.step, self.switch_loss, self.pivot,
             self.doc, self.context, self.loss_word2vec, self.fraction, self.loss_lda, self.loss,
             self.loss_avgs_op, self.optimizer, self.doc_embedding, self.topic_embedding, self.word_embedding,
             self.nce_weights, self.nce_biases, self.merged, *kg) = handles
            if len(kg) > 0:
                self.addtl_features_list = kg[:len(kg) // 2]
                self.feature_lookup = kg[len(kg) // 2:]

    def prior(self):
        doc_prior = dirichlet_likelihood(self.mixture_doc.doc_embedding, alpha=self.alpha)
        feature_prior_created = False
        for i in range(self.n_addtl_features):
            temp_feature_prior = dirichlet_likelihood(self.addtl_features_list[i].doc_embedding, alpha=self.alpha)
            if feature_prior_created:
                feature_prior_created += temp_feature_prior
            else:
                feature_prior_created = True
                feature_prior = temp_feature_prior

        if feature_prior_created:
            # noinspection PyUnboundLocalVariable
            return doc_prior + feature_prior
        else:
            return doc_prior

    def _build_graph(self):
        """
        x = pivot words (int)
        y = context words (int)
        docs = docs at pivot (int)

        :return:
        """
        x = tf.placeholder(tf.int32, shape=[None], name='x_pivot_idxs')
        y = tf.placeholder(tf.int64, shape=[None], name='y_target_idxs')
        docs = tf.placeholder(tf.int32, shape=[None], name='doc_ids')
        addtl_features = tf.placeholder(tf.int32, shape=[self.n_addtl_features, None])
        step = tf.Variable(0, trainable=False, name='global_step')
        switch_loss = tf.Variable(0, trainable=False)
        word_context = tf.nn.embedding_lookup(self.w_embed.embedding, x, name='word_embed_lookup')
        doc_context = self.mixture_doc(doc_ids=docs)
        feature_lookup = []
        for i in range(self.n_addtl_features):
            feature_lookup.append(self.addtl_features_list[i](doc_ids=addtl_features[i]))

        contexts_to_add = feature_lookup
        contexts_to_add.append(word_context)
        contexts_to_add.append(doc_context)
        context = tf.add_n(contexts_to_add, name='context_vector')
        with tf.name_scope('nce_loss'):
            loss_word2vec = self.w_embed(context, y)
            tf.summary.scalar('nce_loss', loss_word2vec)

        with tf.name_scope('lda_loss'):
            fraction = tf.Variable(1, trainable=False, dtype=tf.float32, name='fraction')
            # noinspection PyTypeChecker
            loss_lda = self.lmbda * fraction * self.prior()
            tf.summary.scalar('lda_loss', loss_lda)

        loss = tf.cond(step < switch_loss, lambda: loss_word2vec, lambda: loss_word2vec + loss_lda)
        loss_avgs_op = self.moving_avgs.apply([loss_lda, loss_word2vec, loss])
        with tf.control_dependencies([loss_avgs_op]):
            optimizer = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), self.learning_rate,
                                                        'Adam', name='optimizer')

        self.sess.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        merged = tf.summary.merge_all()
        to_return = [
            x, y, docs, addtl_features, step, switch_loss, word_context, doc_context, context,
            loss_word2vec, fraction, loss_lda, loss, loss_avgs_op, optimizer, self.mixture_doc.doc_embedding,
            self.mixture_doc.topic_embedding, self.w_embed.embedding, self.w_embed.nce_weights,
            self.w_embed.nce_biases, merged
        ]
        if self.n_addtl_features:
            for i in range(self.n_addtl_features):
                to_return.append(self.addtl_features_list[i].doc_embedding)
                to_return.append(self.addtl_features_list[i].topic_embedding)

            to_return.extend(feature_lookup)

        return to_return

    # noinspection PyUnusedLocal
    def train(self, pivot_words, target_words, doc_ids, data_size, n_epochs, context_ids=False,
              idx_to_word=None, switch_loss_epoch=0, save_every=5000, report_every=100):
        """

        :param pivot_words: (array[int]) list of pivot word indices
        :param target_words: (array[int]) list of target word indices
        :param doc_ids: (array[int]) list of doc ids linked to pivot and target words
        :param data_size: (int) number of unique sentences before splitting into pivot/target words
        :param n_epochs: (int) number of epochs to train for
        :param context_ids: (array[int]) list of additional contexts (e.g. zipcode)
        :param idx_to_word:
        :param switch_loss_epoch:
        :param save_every:
        :param report_every:
        :return:
        """
        temp_fraction = self.batch_size * 1.0 / data_size
        self.sess.run(tf.assign(self.fraction, temp_fraction))
        self.n_batches = data_size // self.batch_size
        iters_per_epoch = int(data_size / self.batch_size) + np.ceil(data_size % self.batch_size)
        switch_loss_step = iters_per_epoch * switch_loss_epoch
        self.sess.run(tf.assign(self.switch_loss, switch_loss_step))
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.log_dir + '/', graph=self.sess.graph)
        for epoch in range(n_epochs):
            print('\nEPOCH:', epoch + 1)
            for i in range(self.n_batches + 1):
                batch_start = i * self.batch_size
                batch_end = i * self.batch_size + self.batch_size
                if i < self.n_batches:
                    x_batch = pivot_words[batch_start:batch_end]
                    y_batch = target_words[batch_start:batch_end]
                    docs_batch = doc_ids[batch_start:batch_end]
                    # noinspection PyUnresolvedReferences
                    if type(context_ids) == bool:
                        pass
                    elif context_ids.shape[0] == 1:
                        # noinspection PyUnresolvedReferences
                        features_batch = context_ids[0][batch_start:batch_end]
                        features_batch = np.expand_dims(features_batch, 0)
                    else:
                        features_batch = context_ids[:, batch_start:batch_end]
                else:
                    x_batch = pivot_words[batch_start:]
                    y_batch = target_words[batch_start:]
                    docs_batch = doc_ids[batch_start:]
                    # noinspection PyUnresolvedReferences
                    if type(context_ids) == bool:
                        pass
                    elif context_ids.shape[0] == 1:
                        # noinspection PyUnresolvedReferences
                        features_batch = context_ids[0][batch_start:]
                        features_batch = np.expand_dims(features_batch, 0)
                    else:
                        features_batch = context_ids[:, batch_start:]

                if type(context_ids) == bool:
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.docs: docs_batch}
                else:
                    # noinspection PyUnboundLocalVariable
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.docs: docs_batch,
                                 self.addtl_features: features_batch}

                summary, _, loss, loss_word2vec, loss_lda, step = \
                    self.sess.run([self.merged, self.optimizer, self.loss, self.loss_word2vec, self.loss_lda,
                                   self.step], feed_dict=feed_dict)
                if step > 0 and step % report_every == 0:
                    print('STEP', step, 'LOSS', loss, 'LOSS_WORD2VEC', loss_word2vec, 'LOSS_LDA', loss_lda)

                if step > 0 and step % save_every == 0:
                    idxs = np.arange(self.n_topics)
                    words, sims = self.get_k_closest(idxs, in_type='topic', idx_to_word=idx_to_word, k=11)
                    writer.add_summary(summary, step)
                    writer.flush()
                    writer.close()
                    save_path = saver.save(self.sess, self.log_dir + '/model.ckpt')
                    writer = tf.summary.FileWriter(self.log_dir + '/', graph=self.sess.graph)

        save_path = saver.save(self.sess, self.log_dir + '/model.ckpt')

    def predict(self, pivot_words):
        return self.sess.run([self.context], feed_dict={self.x: pivot_words})

    def compute_normed_embeds(self):
        self.normed_embed_dict = {}
        norm = tf.sqrt(tf.reduce_sum(self.topic_embedding ** 2, 1, keep_dims=True))
        self.normed_embed_dict['topic'] = self.topic_embedding / norm
        norm = tf.sqrt(tf.reduce_sum(self.word_embedding ** 2, 1, keep_dims=True))
        self.normed_embed_dict['word'] = self.word_embedding / norm
        norm = tf.sqrt(tf.reduce_sum(self.doc_embedding ** 2, 1, keep_dims=True))
        self.normed_embed_dict['doc'] = self.doc_embedding / norm
        self.idxs_in = tf.placeholder(tf.int32, shape=[None], name='idxs')
        self.compute_normed = True

    def get_k_closest(self, idxs, in_type='word', vs_type='word', k=10, idx_to_word=None):
        """
        Note: acceptable type pairs:
        word - word
        word - topic
        topic - word
        doc - doc

        :param idxs: (ndarray) numpy array of indexes to check similarity against.
        :param in_type: (str) denotes kind of embedding to check similarity against.
            Options are "word", "doc", and "topic".
        :param vs_type: (str) same as above - denotes kind of embedding that we are
            comparing the in indices to.
        :param k: (int) number of closest examples to get
        :param idx_to_word:
        :return:
        """
        if self.compute_normed is False:
            self.compute_normed_embeds()

        self.batch_array = tf.nn.embedding_lookup(self.normed_embed_dict[in_type], self.idxs_in)
        self.cosine_similarity = tf.matmul(self.batch_array, tf.transpose(self.normed_embed_dict[vs_type], [1, 0]))
        feed_dict = {self.idxs_in: idxs}
        sim, sim_idxs = self.sess.run(tf.nn.top_k(self.cosine_similarity, k=k), feed_dict=feed_dict)
        print('--------- Closest words to given indices ----------')
        if idx_to_word:
            for i, idx in enumerate(idxs):
                if in_type == 'word':
                    in_word = idx_to_word[idx]
                else:
                    in_word = 'Topic ' + str(idx)

                vs_word_list = []
                for vs_i in range(sim_idxs[i].shape[0]):
                    vs_idx = sim_idxs[i][vs_i]
                    vs_word = idx_to_word[vs_idx]
                    vs_word_list.append(vs_word)

                print(in_word, ':', ', '.join(vs_word_list))

        return sim, sim_idxs

    def save_weights_to_file(self, word_embed_path='word_weights', doc_embed_path='doc_weights',
                             topic_embed_path='topic_weights'):
        word_embeds = self.sess.run(self.word_embedding)
        np.save(word_embed_path, word_embeds)
        doc_embeds = self.sess.run(self.doc_embedding)
        np.save(doc_embed_path, doc_embeds)
        topic_embeds = self.sess.run(self.topic_embedding)
        np.save(topic_embed_path, topic_embeds)
