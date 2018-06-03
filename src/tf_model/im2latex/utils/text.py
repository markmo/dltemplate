from collections import Counter
import numpy as np


class Vocab(object):

    def __init__(self, config):
        self.tok2id = None
        self.id2tok = None
        self.n_tok = 0
        self.id_pad = None
        self.id_end = None
        self.id_unk = None
        self.config = config
        self.load_vocab()

    def load_vocab(self):
        special_tokens = [self.config.unk, self.config.pad, self.config.end]
        self.tok2id = load_tok2id(self.config.path_vocab, special_tokens)
        self.id2tok = {idx: tok for tok, idx in self.tok2id.items()}
        self.n_tok = len(self.tok2id)
        self.id_pad = self.tok2id[self.config.pad]
        self.id_end = self.tok2id[self.config.end]
        self.id_unk = self.tok2id[self.config.unk]

    @property
    def form_prepro(self):
        return get_form_prepro(self.tok2id, self.id_unk)


def get_form_prepro(vocab, id_unk):
    """
    Given a vocab, returns a lambda function word -> id

    :param vocab: dict[token] = id
    :param id_unk:
    :return: lambda function formula -> list of ids
    """
    def get_token_id(token):
        return vocab[token] if token in vocab else id_unk

    def f(formula):
        formula = formula.strip().split(' ')
        return [get_token_id(t) for t in formula]

    return f


def load_tok2id(filename, tokens=None):
    """

    :param filename: (string) path to vocab txt file, one word per line
    :param tokens: list of tokens to add to vocab after reading filename
    :return: dict[token] = id
    """
    if tokens is None:
        tokens = []

    tok2id = {}
    with open(filename) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            tok2id[token] = idx

    # add extra tokens
    i = len(tok2id)
    for tok in tokens:
        tok2id[tok] = i
        i += 1

    return tok2id


def build_vocab(datasets, min_count=10):
    """
    Build vocabulary from an iterable of dataset objects

    :param datasets: list of dataset objects
    :param min_count: (int) if token appears less than min_count, then do not include it
    :return: a set of all words in the dataset
    """
    print('Building vocab...')
    c = Counter()
    for dataset in datasets:
        for _, formula in dataset:
            try:
                c.update(formula)
            except Exception as e:
                print(formula)
                raise e

    vocab = [tok for tok, count in c.items() if count >= min_count]
    print('- done. {}/{} tokens added to vocab'.format(len(vocab), len(c)))
    return sorted(vocab)


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Writes one word per line

    :param vocab: iterable that yields word
    :param filename: path to vocab file
    :return: write a word per line
    """
    print('Writing vocab...')
    i = -1
    with open(filename, 'w') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write('{}\n'.format(word))
            else:
                f.write(word)

    print('- done. {} tokens'.format(i + 1))


def pad_batch_formulas(formulas, id_pad, id_end, max_len=None):
    """
    Pad formulas to the max length with id_pad and adds the id_end token
    at the end of each formula.

    :param formulas: list of ints
    :param id_pad:
    :param id_end:
    :param max_len: length of longest formula
    :return: tuple of array shape (batch_size, max_len) of type np.int32, and
             array shape (batch_size) of type np.int32
    """
    if max_len is None:
        max_len = max([len(x) for x in formulas])

    batch_formulas = id_pad * np.ones([len(formulas), max_len + 1], dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula, dtype=np.int32)
        batch_formulas[idx, len(formula)] = id_end
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length


def load_formulas(filename):
    formulas = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()

    print('Loaded {} formulas from {}'.format(len(formulas), filename))
    return formulas
