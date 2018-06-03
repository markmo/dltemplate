import distance
import numpy as np
from scipy.misc import imread
from tf_model.im2latex.utils.general import get_files


def score_dirs(dir_ref, dir_hyp, prepro_img):
    """
    Returns scores from a dir with images

    :param dir_ref: (string)
    :param dir_hyp: (string)
    :param prepro_img: lambda function
    :return: scores (dict)
    """
    img_refs = [f for f in get_files(dir_ref) if f.split('.')[-1] == 'png']
    img_hyps = [f for f in get_files(dir_hyp) if f.split('.')[-1] == 'png']
    em_tot = l_dist_tot = length_tot = m = 0
    for img_name in img_refs:
        img_ref = imread(dir_ref + img_name)
        img_ref = prepro_img(img_ref)
        if img_name in img_hyps:
            img_hyp = imread(dir_hyp + img_name)
            img_hyp = prepro_img(img_hyp)
            l_dist, length = img_edit_distance(img_ref, img_hyp)
        else:
            l_dist = length = img_ref.shape[1]

        l_dist_tot += l_dist
        length_tot += length
        if l_dist < 1:
            em_tot += 1

        m += 1

    # compute scores
    return {
        'EM': em_tot / float(m) if m > 0 else 0,
        'Lev': 1 - l_dist_tot / float(length_tot) if length_tot > 0 else 0
    }


def img_edit_distance(img1, img2):
    """
    Computes Levenshtein distance between two images.

    Slices the images into columns and consider one column as a character.

    :param img1: np array as shape (h, w, 1)
    :param img2: np array as shape (h, w, 1)
    :return: column-wise Levenshtein distance, max length of the two sequences
    """
    # load the image (h, w)
    img1, img2 = img1[:, :, 0], img2[:, :, 0]

    # transpose and convert to 0 or 1
    img1 = np.transpose(img1)
    h1 = img1.shape[1]
    img1 = (img1 <= 128).astype(np.uint8)
    img2 = np.transpose(img2)
    h2 = img2.shape[1]
    img2 = (img2 <= 128).astype(np.uint8)

    # create binaries for each column
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]
    elif h1 > h2:
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item]) + ''.join(['0'] * (h1 - h2)) for item in img2]
    else:
        seq1 = [''.join([str(i) for i in item]) + ''.join(['0'] * (h2 - h1)) for item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]

    # convert each column binary into int
    seq1_int = [int(item, 2) for item in seq1]
    seq2_int = [int(item, 2) for item in seq2]

    # distance
    l_dist = distance.levenshtein(seq1_int, seq2_int)
    length = float(max(len(seq1_int), len(seq2_int)))
    return l_dist, length
