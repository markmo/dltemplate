from argparse import ArgumentParser
from common.load_data import DATA_DIR, load_stack_overflow_questions_dataset, load_word2vec_embeddings
from nlp.duplicate_questions.util import dcg_score, hits_count, prepare_file_from_corpus
from nlp.duplicate_questions.util import rank_candidates, read_corpus
import numpy as np
import os
import subprocess
from tqdm import tqdm


def run(constants):
    embedding_type = constants['embedding_type']
    val_prep_file = DATA_DIR + 'questions/validation_prep.tsv'

    if not os.path.exists(val_prep_file):
        train, val, test = load_stack_overflow_questions_dataset()

        prepare_file_from_corpus(train, DATA_DIR + 'questions/train_prep.tsv')
        prepare_file_from_corpus(val, DATA_DIR + 'questions/validation_prep.tsv')
        prepare_file_from_corpus(test, DATA_DIR + 'questions/test_prep.tsv')

    val_prep = read_corpus(val_prep_file)
    # print('Cleaning text...')
    # val_prep = []
    # for line in tqdm(val):
    #     new_line = [clean_text(q) for q in line]
    #     val_prep.append(new_line)

    if embedding_type == 'word2vec':
        embeddings = load_word2vec_embeddings()
        print('Calculate ranks using word2vec embeddings...')
        ranking = []
        for line in tqdm(val_prep):
            q, *ex = line
            ranks = rank_candidates(q, ex, embeddings)
            ranking.append([r[0] for r in ranks].index(0) + 1)

        for k in [1, 5, 10, 100, 500, 1000]:
            print('DCG@%4d: %.3f | Hits@%4d: %.3f' % (k, dcg_score(ranking, k), k, hits_count(ranking, k)))

    elif embedding_type == 'starspace':
        current_dir = os.path.dirname(os.path.realpath(__file__))
        starspace_embeddings_file = current_dir + '/starspace_embedding.tsv'

        if not os.path.exists(starspace_embeddings_file):
            print('Training starspace embeddings...')
            train_file = DATA_DIR + 'questions/train_prep.tsv'
            cmd = ['starspace', 'train',
                   '-trainFile', f'"{train_file}"',
                   '-model', 'starspace_embedding',
                   '-trainMode', '3',
                   '-adagrad', 'true',
                   '-ngrams', '1',
                   '-epoch', '5',
                   '-dim', '100',
                   '-similarity', '"cosine"',
                   '-minCount', '2',
                   '-verbose', 'true',
                   '-fileFormat', 'labelDoc',
                   '-negSearchLimit', '10',
                   '-lr', '0.05',
                   '-thread', '4'
                   ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in p.stdout:
                print(line)

            p.wait()
            if p.returncode == 0:
                print('All good so far')
            else:
                raise Exception('process returned {}'.format(p.returncode))

        ss_embeddings = {}
        for line in open(starspace_embeddings_file):
            word, *embeds = line.strip().split('\t')
            ss_embeddings[word] = np.array(embeds).astype(np.float32)

        print('Calculate ranks using starspace embeddings...')
        ss_prepared_ranking = []
        for line in tqdm(val_prep):
            q, *ex = line
            ranks = rank_candidates(q, ex, ss_embeddings, dim=100)
            ss_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)

        for k in [1, 5, 10, 100, 500, 1000]:
            print('DCG@%4d: %.3f | Hits@%4d: %.3f' % (k, dcg_score(ss_prepared_ranking, k),
                                                      k, hits_count(ss_prepared_ranking, k)))

    else:
        raise NotImplementedError('Unsupported embedding type {}'.format(embedding_type))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Detect Dup Qs model')
    parser.add_argument('--embed-type', dest='embedding_type', default='starspace', help='embedding type')
    args = parser.parse_args()

    run(vars(args))
