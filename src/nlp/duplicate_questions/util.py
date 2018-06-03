from common.nlp_util import clean_text
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def dcg_score(dup_ranks, k):
    """
    Discounted cumulative gain (DCG) is a measure of ranking quality.

    Using a graded relevance scale of documents in a search-engine result set,
    DCG measures the usefulness, or gain, of a document based on its position
    in the result list. The gain is accumulated from the top of the result list
    to the bottom, with the gain of each result discounted at lower ranks.

    :param dup_ranks:
    :param k:
    :return:
    """
    ranks = np.array(dup_ranks)
    return np.mean(1 / np.log2(1 + ranks) * (ranks <= k))


def hits_count(dup_ranks, k):
    """

    :param dup_ranks: list of ranks for each duplicate (best rank is 1,
                      worst is len(dup_ranks))
    :param k: number of top-ranked elements
    :return: (float)
    """
    return np.mean(np.array(dup_ranks) <= k)


def prepare_file(in_, out_):
    if not os.path.exists(out_):
        print('Preparing {}'.format(out_))
        out = open(out_, 'w')
        for line in tqdm(open(in_, encoding='utf8')):
            line = line.strip().split('\t')
            new_line = [clean_text(q) for q in line]
            print(*new_line, sep='\t', file=out)

        out.close()
    else:
        print('File {} already exists'.format(out_))


def prepare_file_from_corpus(corpus, out_):
    if not os.path.exists(out_):
        print('Preparing {}'.format(out_))
        out = open(out_, 'w')
        for line in tqdm(corpus):
            new_line = [clean_text(q) for q in line]
            print(*new_line, sep='\t', file=out)

        out.close()
    else:
        print('File {} already exists'.format(out_))


def question2vec(question, embeddings, dim=300):
    """

    :param question: (str)
    :param embeddings: (dict) where the key is a word and value is its embedding
    :param dim: size of the representation
    :return: vector representation for the question
    """
    if not question:
        return np.zeros(dim)

    word_vectors = []
    for word in question.split(' '):
        if word in embeddings:
            word_vectors.append(embeddings[word])

    if not word_vectors:
        return np.zeros(dim)

    return np.mean(word_vectors, axis=0)


def rank_candidates(question, candidates, embeddings, dim=300):
    """
    We will use cosine distance to rank candidate questions.

    For example, if the list of candidates was [a, b, c] and the most similar
    is c, then a and b, the function should return a list [(2, c), (0, a), (1, b)].

    :param question: (str)
    :param candidates: list of strings (candidates) we want to rank
    :param embeddings:
    :param dim: dimension of the current embeddings
    :return: a list of pairs (initial position in candidates list, candidate)
    """
    q_embeds = question2vec(question, embeddings, dim)
    q_embeds = np.reshape(q_embeds, (1, dim))
    c_embeds = np.array([question2vec(c, embeddings, dim) for c in candidates])
    similarity = cosine_similarity(q_embeds, c_embeds)
    return [list(enumerate(candidates))[idx] for idx in np.argsort(-similarity[0])]


def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))

    return data
