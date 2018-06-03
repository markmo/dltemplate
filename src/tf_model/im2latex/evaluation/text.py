import distance
import nltk
import numpy as np
from tf_model.im2latex.utils.general import init_dir
from tf_model.im2latex.utils.text import load_formulas


def score_files(path_ref, path_hyp):
    """
    Loads result from file and scores it

    :param path_ref: (string) formulas of reference
    :param path_hyp: (string) formulas of prediction
    :return: scores (dict)
    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # tokenize
    refs = [ref.split(' ') for _, ref in formulas_ref.items()]
    hyps = [hyp.split(' ') for _, hyp in formulas_hyp.items()]

    # score
    return {
        'BLEU-4': bleu_score(refs, hyps) * 100,
        'EM': exact_match_score(refs, hyps) * 100,
        'Edit': edit_distance(refs, hyps) * 100
    }


def exact_match_score(references, hypotheses):
    """
    Computes exact match scores

    :param references: list of list of tokens (one ref)
    :param hypotheses: list of list of tokens (one hypothesis)
    :return: exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """
    Computes bleu score

    :param references: list of list (one ref)
    :param hypotheses: list of list (one hypothesis)
    :return: BLEU-4 score: (float)
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    bleu_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
                                                   weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_4


def edit_distance(references, hypotheses):
    """
    Computes Levenshtein distance between two sequences.

    :param references: list of list of token (one ref)
    :param hypotheses: list of list of token (one hypothesis)
    :return: 1 - Levenshtein distance (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1 - d_leven / len_tot


def truncate_end(list_of_ids, id_end):
    """
    Removes the end of the list starting from the first id_end token.

    :param list_of_ids:
    :param id_end:
    :return:
    """
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_end:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def write_answers(references, hypotheses, rev_vocab, dir_name, id_end):
    """
    Writes text answers in files.

    One file for the reference, one file for each hypothesis.

    :param references: list of list (one reference)
    :param hypotheses: list of list of list (multiple hypotheses)
                       hypotheses[0] is a list of the first hypothesis
                       for all the dataset
    :param rev_vocab: (dict) rev_vocab[idx] = word
    :param dir_name: (string) path to write results
    :param id_end: (int) special id of token that corresponds to the
                   end of sentence
    :return: file_names: list of the created files
    """
    def ids_to_str(ids):
        ids = truncate_end(ids, id_end)
        s = [rev_vocab[idx] for idx in ids]
        return ' '.join(s)

    def write_file(filename, list_of_list):
        with open(filename, 'w') as f:
            for l in list_of_list:
                f.write(ids_to_str(l) + '\n')

    init_dir(dir_name)
    filenames = [dir_name + 'ref.txt']
    write_file(dir_name + 'ref.txt', references)  # one file for the ref
    for i in range(len(hypotheses)):  # one file per hypo
        assert len(references) == len(hypotheses[i])
        write_file(dir_name + 'hyp_{}.txt'.format(i), hypotheses[i])

    return filenames
