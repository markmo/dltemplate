from keras.preprocessing.sequence import skipgrams
import numpy as np
import os
import pandas as pd
import pickle
import pyLDAvis
from sklearn.utils import shuffle
import tensorflow as tf
from tf_model.lda2vec.nlp_pipeline import NlpPipeline


def dirichlet_likelihood(weights, alpha=None):
    """
    Calculate the log likelihood of the observed topic proportions.
    A negative likelihood is more likely than a negative likelihood.

    :param weights: Unnormalized weight vector. The vector
            will be passed through a softmax function that will map the input
            onto a probability simplex.
    :param alpha: (float) The Dirichlet concentration parameter. Alpha
            greater than 1.0 results in very dense topic weights such
            that each document belongs to many topics. Alpha < 1.0 results
            in sparser topic weights. The default is to set alpha to
            1.0 / n_topics, effectively enforcing the prior belief that a
            document belong to every topics at once.
    :return: Output loss variable.
    """
    n_topics = weights.get_shape()[1].value
    if alpha is None:
        alpha = 1.0 / n_topics

    log_proportions = tf.nn.log_softmax(weights)
    loss = (alpha - 1.0) * log_proportions
    return tf.reduce_sum(loss)


def prepare_topics(weights, factors, word_vectors, vocab, temperature=1.0,
                   doc_lengths=None, term_frequency=None, normalize=False):
    """
    Collects a dictionary of word, document and topic distributions.

    :param weights: (array[float]) This must be an array of unnormalized log-odds of document-to-topic
        weights. Shape should be [n_documents, n_topics]
    :param factors: (array[float]) Should be an array of topic vectors. These topic vectors live in the
        same space as word vectors and will be used to find the most similar
        words to each topic. Shape should be [n_topics, n_dim].
    :param word_vectors: (array[float]) This must be a matrix of word vectors. Should be of shape
        [n_words, n_dim].
    :param vocab: (list[str]) These must be the strings for words corresponding to
        indices [0, n_words].
    :param temperature: (float) Used to calculate the log probability of a word. Higher
        temperatures make more rare words more likely.
    :param doc_lengths: (array[int]) An array indicating the number of words in the nth document.
        Must be of shape [n_documents]. Required by pyLDAvis.
    :param term_frequency: (array[int]) An array indicating the overall number of times each token appears
        in the corpus. Must be of shape [n_words]. Required by pyLDAvis.
    :param normalize: (bool) If true, then normalize word vectors
    :return:
        data: (dict) This dictionary is readily consumed by pyLDAVis for topic
        visualization.
    """
    # Map each factor vector to a word
    topic_to_word = []
    assert len(vocab) == word_vectors.shape[0], 'Vocabulary size did not match size of word vectors'
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis=1)[:, None]

    for factor_vector in factors:
        factor_to_word = prob_words(factor_vector, word_vectors, temperature=temperature)
        topic_to_word.append(np.ravel(factor_to_word))

    topic_to_word = np.array(topic_to_word)
    assert np.allclose(np.sum(topic_to_word, axis=1), 1), 'Not all rows in `topic_to_word` sum to 1'

    # Collect document-to-topic distributions, e.g. theta
    doc_to_topic = _softmax_2d(weights)
    assert np.allclose(np.sum(doc_to_topic, axis=1), 1), 'Not all rows in `doc_to_topic` sum to 1'
    data = {
        'topic_term_dists': topic_to_word,
        'doc_topic_dists': doc_to_topic,
        'doc_lengths': doc_lengths,
        'vocab': vocab,
        'term_frequency': term_frequency
    }
    return data


def prob_words(context, vocab, temperature=1.0):
    """This calculates a softmax over the vocabulary as a function of the dot product of context and word."""
    dot = np.dot(vocab, context)
    return _softmax(dot / temperature)


def run_preprocessing(texts, data_dir, run_name, min_freq_threshold=10, max_len=100, bad=None,
                      vectors='en_core_web_lg', n_threads=2, token_type='lemma', only_keep_alpha=False,
                      write_every=10000, merge=False):
    """
    This function abstracts the rest of the preprocessing needed
    to run Lda2Vec in conjunction with the NlpPipeline.

    :param texts: (list[str]) list of text
    :param data_dir: (str) directory where data is held
    :param run_name: (str) Named of directory created to hold preprocessed data
    :param min_freq_threshold: (int, optional) If words occur less frequently than this threshold,
        then purge them from the docs
    :param max_len: (int, optional) Length to pad/cut off sequences
    :param bad: (list|set, optional) words to filter out of dataset
    :param vectors: (str) Name of vectors to load from spacy, e.g. ["en", "en_core_web_sm"]
    :param n_threads: (int, optional) Number of threads used in spacy pipeline
    :param token_type: (str, optional) Type of tokens to keep, one of ["lemma", "lower", "orth"]
    :param only_keep_alpha: (bool, optional) Only keep alpha characters
    :param write_every: (int, optional) Number of documents' data to store before writing cache to skipgrams file
    :param merge: (bool, optional) Merge noun phrases
    :return:
    """
    if bad is None:
        bad = []

    def clean(line):
        return ' '.join(w for w in line.split() if not any(t in w for t in bad))

    # Location for preprocessed data to be stored
    out_path = data_dir + '/' + run_name
    if not os.path.exists(out_path):
        # Make directory to save data in
        os.makedirs(out_path)

        # Remove tokens with these substrings
        bad = set(bad)

        # Preprocess data
        # Convert to unicode (spacy only works with unicode)
        texts = [str(clean(d)) for d in texts]

        # Process the text, no file because we are passing in data directly
        p = NlpPipeline(None, max_len, texts=texts, n_threads=n_threads, only_keep_alpha=only_keep_alpha,
                        token_type=token_type, vectors=vectors, merge=merge)

        # Computes the embed matrix along with other variables
        p.compute_embed_matrix()

        print('Convert data to word2vec indices')
        p.convert_data_to_word2vec_indices()

        print('Trim zeros')
        p.trim_zeros_from_idx_data()

        # Extract the length of each document (needed for pyldaviz)
        doc_lengths = [len(x) for x in p.idx_data]

        # Find the cutoff index
        cutoff = 0
        for i, freq in enumerate(p.freqs):
            if freq < min_freq_threshold:
                cutoff = i
                break
        # Then cutoff the embed matrix
        embed_matrix = p.embed_matrix[:cutoff]

        # Also replace all tokens below cut off in idx_data
        for i in range(len(p.idx_data)):
            p.idx_data[i][p.idx_data[i] > cutoff - 1] = 0

        # Next cut off the frequencies
        freqs = p.freqs[:cutoff]

        print('Convert to skipgrams')
        data = []
        n_examples = p.idx_data.shape[0]

        # Sometimes docs can be less than the required amount for
        # the skipgram function. So we must manually make a counter
        # instead of relying on the enumerated index (i).
        doc_id_counter = 0

        # Additionally, we will keep track of these lower level docs
        # and will purge them later.
        purged_docs = []
        for i, t in enumerate(p.idx_data):
            pairs, _ = skipgrams(t, vocabulary_size=p.vocab_size, window_size=5, shuffle=True, negative_samples=0)

            # Pairs will be 0 if document is less than 2 indexes
            if len(pairs) > 2:
                for pair in pairs:
                    temp_data = pair

                    # Appends doc ID
                    temp_data.append(doc_id_counter)

                    # Appends document index
                    temp_data.append(i)

                    data.append(temp_data)

                doc_id_counter += 1
            else:
                purged_docs.append(i)

            if i // write_every:
                temp_df = pd.DataFrame(data)
                temp_df.to_csv(out_path + '/skipgrams.txt', sep='\t', index=False, header=None, mode='a')
                del temp_df
                data = []

            if i % 500 == 0:
                print('step', i, 'of', n_examples)

        temp_df = pd.DataFrame(data)
        temp_df.to_csv(out_path + '/skipgrams.txt', sep='\t', index=False, header=None, mode='a')
        del temp_df

        # Save embed matrix
        np.save(out_path + '/embed_matrix', embed_matrix)

        # Save the doc lengths to be used later
        # Also purge those that didnt make it into skipgram function
        np.save(out_path + '/doc_lengths', np.delete(doc_lengths, np.array(purged_docs)))

        # Save frequencies to file
        np.save(out_path + '/freqs', freqs)

        # Save vocabulary dictionary to file
        with open(out_path + '/idx_to_word.pkl', 'wb') as f:
            pickle.dump(p.idx_to_word, f)

        with open(out_path + '/word_to_idx.pkl', 'wb') as f:
            pickle.dump(p.word_to_idx, f)


def load_preprocessed_data(data_path, run_name):
    out_path = data_path + '/' + run_name

    # Reload all data
    with open(out_path + '/idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)

    with open(out_path + '/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    freqs = np.load(out_path + '/freqs.npy').tolist()
    embed_matrix = np.load(out_path + '/embed_matrix.npy')
    df = pd.read_csv(out_path + '/skipgrams.txt', sep='\t', header=None)

    # Extract arrays from dataframe
    pivot_ids = df[0].values
    target_ids = df[1].values
    doc_ids = df[2].values

    # Shuffle the data
    pivot_ids, target_ids, doc_ids = shuffle(pivot_ids, target_ids, doc_ids, random_state=0)

    # Hyperparameters
    n_docs = doc_ids.max() + 1
    vocab_size = len(freqs)
    embed_size = embed_matrix.shape[1]

    return (idx_to_word, word_to_idx, freqs, embed_matrix, pivot_ids,
            target_ids, doc_ids, n_docs, vocab_size, embed_size)


def generate_ldavis_data(data_path, run_name, model, idx_to_word, freqs, vocab_size):
    """This function will launch a locally hosted session of pyLDAvis to visualize the results of our model"""
    doc_embed = model.sess.run(model.doc_embedding)
    topic_embed = model.sess.run(model.topic_mebedding)
    word_embed = model.sess.run(model.word_embedding)

    # Extract all unique words in order of index: 0 - vocab_size
    vocabulary = []
    for i in range(vocab_size):
        vocabulary.append(idx_to_word[i])

    # Read document lengths
    doc_lengths = np.load(data_path + '/' + run_name + '/doc_lengths.npy')

    # The `prepare_topics` function is a direct copy from Chris Moody
    vis_data = prepare_topics(doc_embed, topic_embed, word_embed, np.array(vocabulary), doc_lengths=doc_lengths,
                              term_frequency=freqs, normalize=True)
    prepared_vis_data = pyLDAvis.prepare(**vis_data)
    pyLDAvis.show(prepared_vis_data)


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _softmax_2d(x):
    y = x - x.max(axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y
