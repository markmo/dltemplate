from collections import defaultdict
import numpy as np


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
