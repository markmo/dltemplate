from common.load_data import load_word2vec_embeddings
from nlp.duplicate_questions.util import dcg_score, hits_count, rank_candidates, question2vec
import numpy as np
import pytest


@pytest.fixture
def embeddings():
    return load_word2vec_embeddings()


# noinspection PyShadowingNames
def test_word2vec_embeddings(embeddings):
    errmsg = "Something wrong with your embeddings ('%s test isn't correct)"
    most_similar = embeddings.most_similar(positive=['woman', 'king'], negative=['man'])
    assert len(most_similar) > 0 and most_similar[0][0] == 'queen', errmsg % 'Most similar'

    doesnt_match = embeddings.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch'])
    assert doesnt_match == 'cereal', errmsg % "Doesn't match"

    most_similar_to_given = embeddings.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse'])
    assert most_similar_to_given == 'sound', errmsg % 'Most similar to given'


# noinspection PyShadowingNames,SpellCheckingInspection,PyUnresolvedReferences
def test_question2vec(embeddings):
    assert (np.zeros(300) == question2vec('', embeddings, dim=300)).all(), \
        'Must return zeros vector for empty question'

    assert (np.zeros(300) == question2vec('thereisnosuchword', embeddings, dim=300)).all(), \
        'Must return zeros vector for a question consisting of only unknown words'

    assert (embeddings['word'] == question2vec('word', embeddings)).all(), \
        "Embeddings don't match"

    assert ((embeddings['I'] + embeddings['am']) / 2 == question2vec('I am', embeddings)).all(), \
        'Function should calculate the mean of word vectors'

    assert (embeddings['word'] == question2vec('thereisnosuchword word', embeddings)).all(), \
        'Should not include words for which embeddings are unknown'


# noinspection SpellCheckingInspection
def test_hits_count():
    # answers - dup_i
    answers = ['How does the catch keyword determine the type of exception that was thrown']

    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [['How Can I Make These Links Rotate in PHP',
                           'How does the catch keyword determine the type of exception that was thrown',
                           'NSLog array description not memory address',
                           'PECL_HTTP not recognised php ubuntu']]

    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]

    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1, 1, 1]
    for k, correct in enumerate(correct_answers, 1):  # start from index pos 1
        assert np.isclose(hits_count(dup_ranks, k), correct)

    answers = ['How does the catch keyword determine the type of exception that was thrown',
               'Convert Google results object (pure js) to Python object']

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [['How does the catch keyword determine the type of exception that was thrown',
                           'How Can I Make These Links Rotate in PHP'],
                          ['Convert Google results object (pure js) to Python object',
                           'WPF- How to update the changes in list item of a list']]

    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        assert np.isclose(hits_count(dup_ranks, k), correct), \
            'test: both duplicates on the first position in ranked list'

    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [['How Can I Make These Links Rotate in PHP',
                           'How does the catch keyword determine the type of exception that was thrown'],
                          ['Convert Google results object (pure js) to Python object',
                           'WPF- How to update the changes in list item of a list']]

    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, 1]
    for k, correct in enumerate(correct_answers, 1):
        assert np.isclose(hits_count(dup_ranks, k), correct), \
            'test: one candidate on the first position, another — on the second'

    # The third test: both candidates on the second position
    candidates_ranking = [['How Can I Make These Links Rotate in PHP',
                           'How does the catch keyword determine the type of exception that was thrown'],
                          ['WPF- How to update the changes in list item of a list',
                           'Convert Google results object (pure js) to Python object']]

    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1]
    for k, correct in enumerate(correct_answers, 1):
        assert np.isclose(hits_count(dup_ranks, k), correct), 'test: both candidates on the second position'


# noinspection SpellCheckingInspection
def test_dcg_score():
    # answers - dup_i
    answers = ['How does the catch keyword determine the type of exception that was thrown']

    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [['How Can I Make These Links Rotate in PHP',
                           'How does the catch keyword determine the type of exception that was thrown',
                           'NSLog array description not memory address',
                           'PECL_HTTP not recognised php ubuntu']]

    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]

    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1 / np.log2(3), 1 / np.log2(3), 1 / np.log2(3)]
    for k, correct in enumerate(correct_answers, 1):  # start from index pos 1
        assert np.isclose(dcg_score(dup_ranks, k), correct)

    answers = ['How does the catch keyword determine the type of exception that was thrown',
               'Convert Google results object (pure js) to Python object']

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [['How does the catch keyword determine the type of exception that was thrown',
                           'How Can I Make These Links Rotate in PHP'],
                          ['Convert Google results object (pure js) to Python object',
                           'WPF- How to update the changes in list item of a list']]

    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        assert np.isclose(dcg_score(dup_ranks, k), correct), \
            'test: both duplicates on the first position in ranked list'

    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [['How Can I Make These Links Rotate in PHP',
                           'How does the catch keyword determine the type of exception that was thrown'],
                          ['Convert Google results object (pure js) to Python object',
                           'WPF- How to update the changes in list item of a list']]

    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, (1 + (1 / np.log2(3))) / 2]
    for k, correct in enumerate(correct_answers, 1):
        assert np.isclose(dcg_score(dup_ranks, k), correct), \
            'test: one candidate on the first position, another — on the second'

    # The third test: both candidates on the second position
    candidates_ranking = [['How Can I Make These Links Rotate in PHP',
                           'How does the catch keyword determine the type of exception that was thrown'],
                          ['WPF- How to update the changes in list item of a list',
                           'Convert Google results object (pure js) to Python object']]

    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1 / np.log2(3)]
    for k, correct in enumerate(correct_answers, 1):
        assert np.isclose(dcg_score(dup_ranks, k), correct), 'test: both candidates on the second position'


# noinspection PyShadowingNames
def test_rank_candidates(embeddings):
    questions = ['converting string to list', 'Sending array via Ajax fails']

    candidates = [['Convert Google results object (pure js) to Python object',
                   'C# create cookie from string and send it',
                   'How to use jQuery AJAX for an outside domain?'],
                  ['Getting all list items of an unordered list in PHP',
                   'WPF- How to update the changes in list item of a list',
                   'select2 not displaying search results']]

    results = [[(1, 'C# create cookie from string and send it'),
                (0, 'Convert Google results object (pure js) to Python object'),
                (2, 'How to use jQuery AJAX for an outside domain?')],
               [(0, 'Getting all list items of an unordered list in PHP'),
                (2, 'select2 not displaying search results'),
                (1, 'WPF- How to update the changes in list item of a list')]]

    for q, cs, result in zip(questions, candidates, results):
        ranks = rank_candidates(q, cs, embeddings, dim=300)
        assert np.all(ranks == result)
