from argparse import ArgumentParser
from common.load_data import DATA_DIR
from common.model_util import load_hyperparams, merge_dict
import pandas as pd
from pathlib import Path
from tf_model.lda2vec.model_setup import Lda2vecModel
from tf_model.lda2vec.preprocessor import Preprocessor
from tf_model.lda2vec.util import generate_ldavis_data, load_preprocessed_data
# from tf_model.lda2vec.util_deprecated import run_preprocessing
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# DATA_PATH = Path(DATA_DIR) / 'onesource' / 'combine_text.txt'
DATA_PATH = Path(DATA_DIR) / '20_newsgroups' / '20_newsgroups.txt'


def load_data():
    texts = []
    with open(DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            texts.append(line.rstrip('\n'))

    return texts


def run(constant_overwrites):
    this_dir = Path(__file__).parent
    config_path = this_dir / 'hyperparams.yml'
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    print('Loading data')
    texts = load_data()
    print('texts length:', len(texts))
    df = pd.DataFrame(texts, columns=['texts'], dtype=str)
    data_dir = this_dir / 'data'
    clean_data_dir = data_dir / 'clean'

    # run_name = 'run_' + time.strftime('%Y%m%d-%H%M%S')
    # bad = {'ax>', '`@("', '---', '===', '^^^', 'AX>', 'GIZ', '--'}

    if not clean_data_dir.exists():
        p = Preprocessor(df, 'texts', max_features=30000)
        p.preprocess()
        embed_matrix = p.load_glove(Path(DATA_DIR) / 'glove' / 'glove.6B.300d.txt')
        p.save_data(clean_data_dir, embed_mat=embed_matrix)
        idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids = load_preprocessed_data(clean_data_dir)
    else:
        (idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids,
         embed_matrix) = load_preprocessed_data(clean_data_dir, load_embed_mat=True)

    # will skip if run dir already exists
    # run_preprocessing(texts, data_dir, run_name, max_len=10000, bad=bad)
    # , vectors='en_core_web_sm', write_every=10000)

    # Load preprocessed data
    # (idx_to_word, word_to_idx, freqs, embed_matrix, pivot_ids,
    #  target_ids, doc_ids, n_docs, vocab_size, embed_size) = load_preprocessed_data(data_dir, run_name)

    n_docs = doc_ids.max() + 1
    vocab_size = len(freqs)
    # embed_size = constants['embed_size']
    embed_size = embed_matrix.shape[1]
    model = Lda2vecModel(n_docs, vocab_size, constants['n_topics'],
                         freqs=freqs, load_embeds=True, pretrained_embeddings=embed_matrix,
                         embed_size=embed_size)
    model.train(pivot_ids, target_ids, doc_ids, len(pivot_ids), constants['n_epochs'],
                idx_to_word=idx_to_word, switch_loss_epoch=5)

    if constants['viz']:
        generate_ldavis_data(clean_data_dir, model, idx_to_word, freqs, vocab_size)
        # generate_ldavis_data(data_dir, run_name, model, idx_to_word, freqs, vocab_size)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run LDA2Vec model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--topics', dest='n_topics', type=int, help='number topics')
    parser.add_argument('--viz', dest='viz', help='visualize results', action='store_true')
    parser.set_defaults(viz=False)
    args = parser.parse_args()

    run(vars(args))
