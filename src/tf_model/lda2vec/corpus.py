from collections import defaultdict
import numpy as np
import pandas as pd

try:
    # noinspection PyUnresolvedReferences
    from pyxdameraulevenshtein import damerau_levenshtein_distance_withNPArray
except ImportError:
    pass


class Corpus(object):

    _keys_frequency = None

    def __init__(self, oov=-1, skip=-2):
        """
        The Corpus helps with tasks involving integer representations of
        words. This object is used to filter, subsample, and convert loose
        word indices into compact word indices.

        "Loose" word arrays are word indices given by a tokenizer. The word
        index is not necessarily representative of words' frequency rank, and
        so loose arrays tend to have gaps of unused indices, which can make
        models less memory efficient. As a result, this class helps convert
        a 'loose' array into a 'compact' one where the most common words have
        low indices, and the most infrequent have high indices.

        Corpus maintains a count of how many of each word it has seen so
        that it can later selectively filter frequent or rare words. However,
        since word popularity rank could change with incoming data the word
        index count must be updated fully and `self.finalize()` must be called
        before any filtering and sub-sampling operations can happen.

        Examples
        --------
        >>> corpus = Corpus()
        >>> words_raw = np.random.randint(100, size=25)
        >>> corpus.update_word_count(words_raw)
        >>> corpus.finalize()
        >>> words_compact = corpus.to_compact(words_raw)
        >>> words_pruned = corpus.filter_count(words_compact, min_count=2)
        >>> # words_sub = corpus.subsample_frequent(words_pruned, thresh=1e-5)
        >>> words_loose = corpus.to_loose(words_pruned)
        >>> not_oov = words_loose > -1
        >>> np.all(words_loose[not_oov] == words_raw[not_oov])
        True

        :param oov: (int, default=-1) Token index to replace whenever we encounter
               a rare or unseen word. Instead of skipping the token, we mark as an
               out-of-vocabulary (OOV) word.
        :param skip: (int, default=-2) Token index to replace whenever we want to
               skip the current frame. Particularly useful when sub-sampling words
               or when padding a sentence.
        """
        self.counts_loose = defaultdict(int)
        self._finalized = False
        self.specials = dict(oov=oov, skip=skip)
        self.keys_loose = None
        self.keys_counts = None
        self.keys_compact = None
        self.loose_to_compact = None
        self.compact_to_loose = None
        self.specials_to_compact = None
        self.compact_to_special = None
        self._finalized = False

    @property
    def n_specials(self):
        return len(self.specials)

    def update_word_count(self, loose_array):
        """
        Update the corpus word counts given a loose array of word indices.
        Can be called multiple times, but once `finalize` is called the word
        counts cannot be updated.

        Examples
        --------
        >>> corpus = Corpus()
        >>> corpus.update_word_count(np.arange(10))
        >>> corpus.update_word_count(np.arange(8))
        >>> corpus.counts_loose[0]
        2
        >>> corpus.counts_loose[9]
        1

        :param loose_array: (array[int]) array of word indices
        :return:
        """
        self._check_unfinalized()
        uniques, counts = np.unique(np.ravel(loose_array))
        assert uniques.min() >= min(self.specials.values()), \
            'Loose arrays cannot have elements below the values of special tokens as these indices are reserved'
        for k, v in zip(uniques, counts):
            self.counts_loose[k] += v

    def _loose_keys_ordered(self):
        """Get the loose keys in order of decreasing frequency"""
        loose_counts = sorted(self.counts_loose.items(), key=lambda x: x[1], reverse=True)
        keys = np.array(loose_counts)[:, 0]
        counts = np.array(loose_counts)[:, 1]
        order = np.argsort(counts)[::-1].astype('int32')
        keys, counts = keys[order], counts[order]

        # Add in the specials as a prefix to the other keys
        specials = np.sort(self.specials.values())
        keys = np.concatenate([specials, keys])
        empty = np.zeros(len(specials), dtype='int32')
        counts = np.concatenate([empty, counts])
        n_keys = keys.shape[0]
        assert counts.min() >= 0
        return keys, counts, n_keys

    def finalize(self):
        """
        Call `finalize` once done updating word counts. This means that the
        object will no longer accept new word count data, but the loose
        to compact index mapping can be computed. This frees the object to
        filter, subsample, and compact incoming word arrays.

        Examples
        --------
        >>> corpus = Corpus()
        >>> # We'll update the word counts, making sure that word index 2
        >>> # is the most common word index.
        >>> corpus.update_word_count(np.arange(1) + 2)
        >>> corpus.update_word_count(np.arange(3) + 2)
        >>> corpus.update_word_count(np.arange(10) + 2)
        >>> corpus.update_word_count(np.arange(8) + 2)
        >>> corpus.counts_loose[2]
        4
        >>> # The corpus has not been finalized yet, and so the compact mapping
        >>> # has not yet been computed.
        >>> corpus.keys_counts[0]
        Traceback (most recent call last):
            ...
        AttributeError: Corpus instance has no attribute 'keys_counts'
        >>> corpus.finalize()
        >>> corpus.n_specials
        2
        >>> # The special tokens are mapped to the first compact indices
        >>> corpus.compact_to_loose[0]
        -2
        >>> corpus.compact_to_loose[0] == corpus.specials['skip']
        True
        >>> corpus.compact_to_loose[1] == corpus.specials['out_of_vocabulary']
        True
        >>> corpus.compact_to_loose[2]  # Most popular token is mapped next
        2
        >>> corpus.loose_to_compact[3]  # 2nd most popular token is mapped next
        4
        >>> first_non_special = corpus.n_specials
        >>> corpus.keys_counts[first_non_special] # First normal token
        4

        :return:
        """
        # Return the loose keys and counts in descending count order
        # so that the counts array is already in compact order
        self.keys_loose, self.keys_counts, n_keys = self._loose_keys_ordered()
        self.keys_compact = np.arange(n_keys).astype('int32')
        self.loose_to_compact = {l: c for l, c in zip(self.keys_loose, self.keys_compact)}
        self.compact_to_loose = {c: l for l, c in self.loose_to_compact.items()}
        self.specials_to_compact = {s: self.loose_to_compact[i] for s, i in self.specials.items()}
        self.compact_to_special = {c: s for c, s in self.specials_to_compact.items()}
        self._finalized = True

    @property
    def keys_frequency(self):
        if self._keys_frequency is None:
            self._keys_frequency = self.keys_counts * 1.0 / np.sum(self.keys_counts)

        return self._keys_frequency

    def _check_finalized(self):
        assert self._finalized, 'self.finalized() must be called before any other array ops'

    def _check_unfinalized(self):
        assert not self._finalized, 'Cannot update word counts after self.finalized() has been called'

    def filter_count(self, words_compact, min_count=15, max_count=0,
                     max_replacement=None, min_replacement=None):
        """
        Replace word indices below min_count with the pad index.

        Examples
        --------
        >>> corpus = Corpus()
        >>> # Make 1000 word indices with index < 100 and
        >>> # update the word counts.
        >>> word_indices = np.random.randint(100, size=1000)
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()  # any word indices above 99 will be filtered
        >>> # Now create a new text, but with some indices above 100
        >>> word_indices = np.random.randint(200, size=1000)
        >>> word_indices.max() < 100
        False
        >>> # Remove words that have never appeared in the original corpus.
        >>> filtered = corpus.filter_count(word_indices, min_count=1)
        >>> filtered.max() < 100
        True
        >>> # We can also remove highly frequent words.
        >>> filtered = corpus.filter_count(word_indices, max_count=2)
        >>> len(np.unique(word_indices)) > len(np.unique(filtered))
        True

        :param words_compact: (array[int]) Source array whose values will be replaced. This is assumed to
            already be converted into a compact array with `to_compact`.
        :param min_count: (int) Replace words less frequently occurring than this count. This
            defines the threshold for what words are very rare.
        :param max_count: (int) Replace words occurring more frequently than this count. This
            defines the threshold for very frequent words.
        :param max_replacement: (int, default=<oov_idx>) Replace words greater than max_count with this.
        :param min_replacement: (int, default=<oov_idx>) Replace words less than min_count with this.
        :return:
        """
        self._check_finalized()
        ret = words_compact.copy()
        if min_replacement is None:
            min_replacement = self.specials_to_compact['oov']

        if max_replacement is None:
            max_replacement = self.specials_to_compact['oov']

        not_specials = np.ones(self.keys_counts.shape[0], dtype='bool')
        not_specials[:self.n_specials] = False
        if min_count:
            # Find first index with count less than min_count
            min_idx = np.argmax(not_specials & (self.keys_counts < min_count))

            # Replace all indices greater than min_idx
            ret[ret > min_idx] = min_replacement

        if max_count:
            # Find first index with count less than max_count
            max_idx = np.argmax(not_specials & (self.keys_counts < max_count))

            # Replace all indices less than max_idx
            ret[ret < max_idx] = max_replacement

        return ret

    # noinspection SpellCheckingInspection
    def subsample_frequent(self, words_compact, threshold=1e-5):
        """
        Subsample the most frequent words. This aggressively
        replaces words with frequencies higher than `threshold`. Words
        are replaced with the out-of-vocabulary token.

        Words will be replaced with probability as a function of their
        frequency in the training corpus:

        .. math::
            p(w) = 1.0 - \sqrt{threshold\over f(w)}

        Examples
        --------
        >>> corpus = Corpus()
        >>> word_indices = (np.random.power(5.0, size=1000) * 100).astype('i')
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> compact = corpus.to_compact(word_indices)
        >>> sampled = corpus.subsample_frequent(compact, threshold=1e-2)
        >>> skip = corpus.specials_to_compact['skip']
        >>> np.sum(compact == skip)  # No skips in the compact tokens
        0
        >>> np.sum(sampled == skip) > 0  # Many skips in the sampled tokens
        True

        .. [1] Distributed Representations of Words and Phrases and
               their Compositionality. Mikolov, Tomas and Sutskever, Ilya
               and Chen, Kai and Corrado, Greg S and Dean, Jeff
               Advances in Neural Information Processing Systems 26

        :param words_compact: (array[int]) The input array to subsample.
        :param threshold: (float in [0, 1]) Words with frequencies higher
            than this will be increasingly sub-sampled.
        :return:
        """
        self._check_finalized()
        freq = self.keys_frequency + 1e-10
        pw = 1.0 - (np.sqrt(threshold / freq) + threshold / freq)
        prob = fast_replace(words_compact, self.keys_compact, pw)
        draw = np.random.uniform(size=prob.shape)
        ret = words_compact.copy()

        # If probability greater than draw, skip the word
        ret[prob > draw] = self.specials_to_compact['skip']
        return ret

    # noinspection SpellCheckingInspection
    def to_compact(self, words_loose):
        """
        Convert a loose word index matrix to a compact array using
        a fixed loose to dense mapping. Out of vocabulary word indices
        will be replaced by the out-of-vocabulary index. The most common
        index will be mapped to 0, the next most common to 1, and so on.

        Examples
        --------
        >>> corpus = Corpus()
        >>> word_indices = np.random.randint(100, size=1000)
        >>> n_words = len(np.unique(word_indices))
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> word_compact = corpus.to_compact(word_indices)
        >>> # The most common word in the training set will be mapped to be
        >>> # right after all the special tokens, so 2 in this case.
        >>> np.argmax(np.bincount(word_compact)) == 2
        True
        >>> most_common = np.argmax(np.bincount(word_indices))
        >>> corpus.loose_to_compact[most_common] == 2
        True
        >>> # Out of vocabulary indices will be mapped to 1
        >>> word_indices = np.random.randint(150, size=1000)
        >>> word_compact_oov = corpus.to_compact(word_indices)
        >>> oov = corpus.specials_to_compact['out_of_vocabulary']
        >>> oov
        1
        >>> oov in word_compact
        False
        >>> oov in word_compact_oov
        True

        :param words_loose: (array[int]) Input loose word array to be converted into a compact array.
        :return:
        """
        self._check_finalized()
        keys = self.keys_loose
        reps = self.keys_compact
        uniques = np.unique(words_loose)

        # Find the out of vocab indices
        oov = np.setdiff1d(uniques, keys, assume_unique=True)
        oov_token = self.specials_to_compact['oov']
        keys = np.concatenate([keys, oov])
        reps = np.concatenate([reps, np.zeros_like(oov) + oov_token])
        compact = fast_replace(words_loose, keys, reps)
        assert compact.min() >= 0, 'Error: all compact indices should be non-negative'
        return compact

    def to_loose(self, words_compact):
        """
        Convert a compacted array back into a loose array.

        Examples
        --------
        >>> corpus = Corpus()
        >>> word_indices = np.random.randint(100, size=1000)
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> word_compact = corpus.to_compact(word_indices)
        >>> word_loose = corpus.to_loose(word_compact)
        >>> np.all(word_loose == word_indices)
        True

        :param words_compact: (array[int]) Input compacted word array to be converted into a loose array.
        :return:
        """
        self._check_finalized()
        uniques = np.unique(words_compact)

        # Find the out of vocab indices
        oov = np.setdiff1d(uniques, self.keys_compact, assume_unique=True)
        assert np.all(oov < 0, ('Found keys in `word_compact` not present in the training corpus. '
                                'Is this actually a compacted array?'))
        loose = fast_replace(words_compact, self.keys_compact, self.keys_loose)
        return loose

    def compact_to_flat(self, words_compact, *components):
        """
        Ravel a 2D compact array of documents (rows) and word
        positions (columns) into a 1D array of words. Leave out special
        tokens and ravel the component arrays in the same fashion.

        Examples
        --------
        >>> corpus = Corpus()
        >>> word_indices = np.random.randint(100, size=1000)
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> doc_texts = np.arange(8).reshape((2, 4))
        >>> doc_texts[:, -1] = -2  # Mark as skips
        >>> doc_ids = np.arange(2)
        >>> compact = corpus.to_compact(doc_texts)
        >>> oov = corpus.specials_to_compact['out_of_vocabulary']
        >>> compact[1, 3] = oov  # Mark the last word as OOV
        >>> flat = corpus.compact_to_flat(compact)
        >>> flat.shape[0] == 6  # 2 skips were dropped from 8 words
        True
        >>> flat[-1] == corpus.loose_to_compact[doc_texts[1, 2]]
        True
        >>> flat, (flat_id,) = corpus.compact_to_flat(compact, doc_ids)
        >>> flat_id
        array([0, 0, 0, 1, 1, 1])

        :param words_compact: (array[int]) Array of word indices in documents. Has shape (n_docs, max_len)
        :param components: (list[array]) A list of arrays detailing per-document properties. Each array
            must n_docs long.
        :return:
            flat: (array[int]) An array of all words unravelled into a 1D shape
            components: (list[array]) Each array here is also unravelled into the same shape
        """
        self._check_finalized()
        n_docs = words_compact.shape[0]
        max_len = words_compact.shape[1]
        idx = words_compact > self.n_specials
        components_raveled = []
        for component in components:
            raveled = np.tile(component[:, None], max_len)[idx]
            components_raveled.append(raveled)
            assert len(component) == n_docs, 'Length of each component must much `word_compact` size'

        if len(components_raveled) == 0:
            return words_compact[idx]
        else:
            return words_compact[idx], components_raveled

    def word_list(self, vocab, max_compact_index=None, oov_token='<oov>'):
        """
        Translate compact keys back into string representations for a word.

        Examples
        --------
        >>> vocab = {0: 'But', 1: 'the', 2: 'night', 3: 'was', 4: 'warm'}
        >>> word_indices = np.zeros(50).astype('int32')
        >>> word_indices[:25] = 0  # 'But' shows 25 times
        >>> word_indices[25:35] = 1  # 'the' is in 10 times
        >>> word_indices[40:46] = 2  # 'night' is in 6 times
        >>> word_indices[46:49] = 3  # 'was' is in 3 times
        >>> word_indices[49:] = 4  # 'warm' in in 2 times
        >>> corpus = Corpus()
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> # Build a vocabulary of word indices
        >>> corpus.word_list(vocab)
        ['skip', 'out_of_vocabulary', 'But', 'the', 'night', 'was', 'warm']

        :param vocab: (dict) The vocab object has loose indices as keys and word strings as
            values.
        :param max_compact_index: (int) Only return words up to this index. If None, defaults to the number
            of compact indices available.
        :param oov_token: (str) Returns this string if a compact index does not have a word in the
            vocab dictionary provided.
        :return:
            word_list: (list) A list of strings corresponding to word indices
            zero to `max_compact_index`
        """
        # Translate the compact keys into strings
        oov = self.specials['oov']
        words = []
        if max_compact_index is None:
            max_compact_index = self.keys_compact.shape[0]

        index_to_special = {i: s for s, i in self.specials.items()}
        for compact_index in range(max_compact_index):
            loose_index = self.compact_to_loose.get(compact_index, oov)
            special = index_to_special.get(loose_index, oov_token)
            string = vocab.get(loose_index, special)
            words.append(string)

        return words

    # noinspection PyPep8,PyUnresolvedReferences,SpellCheckingInspection
    def compact_word_vectors(self, vocab, filename=None, array=None, top=20000):
        """
        Retrieve pretrained word vectors for our vocabulary.
        The returned word array has row indices corresponding to the
        compact index of a word, and columns corresponding to the word
        vector.

        Examples
        --------
        >>> import numpy.linalg as nl
        >>> vocab = {19: 'shuttle', 5: 'astronomy', 7: 'cold', 3: 'hot'}
        >>> word_indices = np.zeros(50).astype('int32')
        >>> word_indices[:25] = 19  # 'Shuttle' shows 25 times
        >>> word_indices[25:35] = 5  # 'astronomy' is in 10 times
        >>> word_indices[40:46] = 7  # 'cold' is in 6 times
        >>> word_indices[46:] = 3  # 'hot' is in 3 times
        >>> corpus = Corpus()
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> v, s, f = corpus.compact_word_vectors(vocab)
        >>> sim = lambda x, y: np.dot(x, y) / nl.norm(x) / nl.norm(y)
        >>> vocab[corpus.compact_to_loose[2]]
        'shuttle'
        >>> vocab[corpus.compact_to_loose[3]]
        'astronomy'
        >>> vocab[corpus.compact_to_loose[4]]
        'cold'
        >>> sim_shuttle_astro = sim(v[2, :], v[3, :])
        >>> sim_shuttle_cold = sim(v[2, :], v[4, :])
        >>> sim_shuttle_astro > sim_shuttle_cold
        True

        :param vocab: (dict) Dictionary where keys are the loose index, and values are
            the word string.
        :param filename: (str) Filename for word2vec vectors via gensim.
        :param array:
        :param top:
        :return:
            data: (ndarray[float]) Array such that data[compact_index, :] = word_vector
        """
        n_words = len(self.compact_to_loose)

        from gensim.models.word2vec import Word2Vec

        model = Word2Vec.load_word2vec_format(filename, binary=True)
        n_dim = model.syn0.shape[1]
        data = np.random.normal(size=(n_words, n_dim)).astype('float32')
        data -= data.mean()
        data += model.syn0.mean()
        data /= data.std()
        data *= model.syn0.std()
        if array is not None:
            data = array
            # n_words = data.shape[0]

        keys_raw = model.vocab.keys()
        keys = [s.encode('ascii', 'ignore') for s in keys_raw]
        lens = [len(s) for s in model.vocab.keys()]
        choices = np.array(keys, dtype='S')
        lengths = np.array(lens, dtype='int32')
        s, f = 0, 0
        rep0 = lambda w: w
        rep1 = lambda w: w.replace(' ', '_')
        rep2 = lambda w: w.title().replace(' ', '_')
        reps = [rep0, rep1, rep2]
        for compact in np.arange(top):
            loose = self.compact_to_loose.get(compact, None)
            if loose is None:
                continue

            word = vocab.get(loose, None)
            if word is None:
                continue

            word = word.strip()
            vector = None
            for rep in reps:
                clean = rep(word)
                if clean in model.vocab:
                    vector = model[clean]
                    break

            if vector is None:
                try:
                    word = str(word)
                    idx = lengths >= len(word) - 3
                    idx &= lengths <= len(word) + 3
                    sel = choices[idx]
                    d = damerau_levenshtein_distance_withNPArray(word, sel)
                    choice = np.array(keys_raw)[idx][np.argmin(d)]
                    vector = model[choice]
                    print(compact, word, ' --> ', choice)
                except IndexError:
                    pass

            if vector is None:
                f += 1
                continue

            s += 1
            data[compact, :] = vector[:]

        return data, s, f

    # noinspection PyMethodMayBeStatic,SpellCheckingInspection
    def compact_to_bow(self, word_compact, max_compact_index=None):
        """
        Given a 2D array of compact indices, return the bag of words
        representation where the column is the word index, row is the document
        index, and value is the number of times that word appears in the
        document.

        Examples
        --------
        >>> import numpy.linalg as nl
        >>> vocab = {19: 'shuttle', 5: 'astronomy', 7: 'cold', 3: 'hot'}
        >>> word_indices = np.zeros(50).astype('int32')
        >>> word_indices[:25] = 19  # 'Shuttle' shows 25 times
        >>> word_indices[25:35] = 5  # 'astronomy' is in 10 times
        >>> word_indices[40:46] = 7  # 'cold' is in 6 times
        >>> word_indices[46:] = 3  # 'hot' is in 3 times
        >>> corpus = Corpus()
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()
        >>> v = corpus.compact_to_bow(word_indices)
        >>> len(v)
        20
        >>> v[:6]
        array([ 5,  0,  0,  4,  0, 10])
        >>> v[19]
        25
        >>> v.sum()
        50
        >>> words = [[0, 0, 0, 3, 4], [1, 1, 1, 4, 5]]
        >>> words = np.array(words)
        >>> bow = corpus.compact_to_bow(words)
        >>> bow.shape
        (2, 6)

        :param word_compact:
        :param max_compact_index:
        :return:
        """
        if max_compact_index is None:
            max_compact_index = word_compact.max()

        def bin_count(x):
            return np.bincount(x, minlength=max_compact_index + 1)

        axis = len(word_compact.shape) - 1
        bow = np.apply_along_axis(bin_count, axis, word_compact)
        return bow

    # noinspection PyMethodMayBeStatic,PyPep8
    def compact_to_cooccurrence(self, words_compact, indices, window_size=10):
        """
        From an array of compact tokens and aligned array of document indices,
        compute (word, word, document) co-occurrences within a moving window.

        Examples
        --------
        >>> compact = np.array([0, 1, 1, 1, 2, 2, 3, 0])
        >>> doc_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        >>> corpus = Corpus()
        >>> counts = corpus.compact_to_cooccurrence(compact, {'doc': doc_idx})
        >>> counts.counts.sum()
        24
        >>> counts.query('doc == 0').counts.values
        array([3, 3, 6])
        >>> compact = np.array([0, 1, 1, 1, 2, 2, 3, 0])
        >>> doc_idx = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        >>> corpus = Corpus()
        >>> counts = corpus.compact_to_cooccurrence(compact, {'doc': doc_idx})
        >>> counts.counts.sum()
        14
        >>> counts.query('doc == 0').word_index_x.values
        array([0, 1, 1])
        >>> counts.query('doc == 0').word_index_y.values
        array([1, 0, 1])
        >>> counts.query('doc == 0').counts.values
        array([2, 2, 2])
        >>> counts.query('doc == 1').counts.values
        array([1, 1])

        :param words_compact: (array[int]) Sequence of tokens.
        :param indices: (dict of int arrays) Each array in this dictionary should represent the document index it
            came from.
        :param window_size: (int) Indicates the moving window size around which all co-occurrences will
            be computed.
        :return:
            counts: (DataFrame) a DataFrame with two columns for word index A and B,
            one extra column for each document index, and a final column for counts
            in that key.
        """
        # noinspection PyUnresolvedReferences
        tokens = pd.DataFrame(dict(word_index=words_compact)).reset_index()
        for name, index in indices.items():
            tokens[name] = index

        a, b = tokens.copy(), tokens.copy()
        mask = lambda x: np.prod([x[k + '_x'] == x[k + '_y'] for k in indices.keys()], axis=0)
        group_keys = ['word_index_x', 'word_index_y', ]
        group_keys += [k + '_x' for k in indices.keys()]
        total = []
        a['frame'] = a['index'].copy()
        for frame in range(-window_size, window_size + 1):
            if frame == 0:
                continue

            b['frame'] = b['index'] + frame
            matches = (
                a.merge(b, on='frame')
                 .assign(same_doc=mask)
                 .pipe(lambda df: df[df['same_doc'] == 1])
                 .groupby(group_keys)['frame']
                 .count()
                 .reset_index()
            )
            total.append(matches)

        # noinspection PyUnresolvedReferences
        counts = (
            pd.concat(total)
              .groupby(group_keys)['frame']
              .sum()
              .reset_index()
              .rename(columns={k + '_x': k for k in indices.keys()})
              .rename(columns=dict(frame='counts'))
        )
        return counts


def fast_replace(data, keys, values, skip_checks=False):
    """
    Do a search-and-replace in `data` array.

    Examples
    --------
    >>> fast_replace(np.arange(5), np.arange(5), np.arange(5)[::-1])
    array([4, 3, 2, 1, 0])

    :param data: (array[int])
    :param keys: (array[int]) array of keys inside `data` to be replaced
    :param values: (array[int]) array of values that replace the `keys` array
    :param skip_checks: (bool, default=False, optional) skip sanity checking the input
    :return:
    """
    assert np.allclose(keys.shape, values.shape)
    if not skip_checks:
        assert data.max() <= keys.max(), 'data has elements not in keys'

    sdx = np.argsort(keys)
    keys, values = keys[sdx], values[sdx]
    idx = np.digitize(data, keys, right=True)
    return values[idx]
