from argparse import ArgumentParser
import collections
from common.load_data import DATA_DIR
from common.util import load_hyperparams, merge_dict
from fastai import lm_rnn
import fastai.core as fastai
import fastai.dataloader as dataloader
import fastai.dataset as fdata
import fastai.metrics as fmetrics
import fastai.text as ftext
import functools
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from text_classification_benchmarks.fastai.util import get_all, preprocess_csv
import torch

logging.getLogger().setLevel(logging.INFO)

CHUNKSIZE = 24000
MAX_VOCAB = 60000
MIN_FREQ = 2

# noinspection SpellCheckingInspection
PATH = Path(DATA_DIR + 'text_classification/aclImdb/')
# noinspection SpellCheckingInspection
LM_PATH = Path('data/imdb_lm/')
# noinspection SpellCheckingInspection
CLAS_PATH = Path('data/imdb_clas/')


def print_results(n, p, r):
    print('N\t' + str(n))
    print('P@{}\t{:.3f}'.format(1, p))
    print('R@{}\t{:.3f}'.format(1, r))


# noinspection SpellCheckingInspection
def run(constant_overwrites):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    # data subdir expected to exist
    LM_PATH.mkdir(exist_ok=True)
    (LM_PATH/'tmp').mkdir(exist_ok=True)
    CLAS_PATH.mkdir(exist_ok=True)
    (CLAS_PATH / 'tmp').mkdir(exist_ok=True)

    data_path = dir_path + '/train.csv'
    if not os.path.exists(data_path):
        train_df, val_df, test_df, x_train, y_train, x_val, y_val, x_test, y_test, classes = preprocess_csv()
    else:
        train_df = pd.read_csv(dir_path + '/train.csv', header=None, chunksize=CHUNKSIZE)
        # x_train, y_train = train_df[0].values, train_df[1].values
        val_df = pd.read_csv(dir_path + '/val.csv', header=None, chunksize=CHUNKSIZE)
        # x_val, y_val = val_df[0].values, val_df[1].values
        # test_df = pd.read_csv(dir_path + '/test.csv', header=None, chunksize=CHUNKSIZE)
        # x_test, y_test = test_df[0].values, test_df[1].values
        # classes = np.genfromtxt(dir_path + '/classes.txt', dtype=str)

    # print('Counts x_train: {}, y_train: {}, x_val: {}, y_val: {}, x_test: {}, y_test: {}, classes: {}'
    #       .format(len(x_train), len(y_train), len(x_val), len(y_val), len(x_test), len(y_test), len(classes)))

    if constants['train_lm']:
        logging.info('Training LM...')
        if (LM_PATH / 'tmp' / 'tok_train.npy').exists():
            logging.info('Loading tokens...')
            tok_train = np.load(LM_PATH / 'tmp' / 'tok_train.npy')
            tok_val = np.load(LM_PATH / 'tmp' / 'tok_val.npy')
        else:
            logging.info('Get tokens...')
            tok_train, labels_train = get_all(train_df, 1)
            tok_val, labels_val = get_all(val_df, 1)
            np.save(LM_PATH / 'tmp' / 'tok_train.npy', tok_train)
            np.save(LM_PATH / 'tmp' / 'tok_val.npy', tok_val)

        if (LM_PATH / 'tmp' / 'itos.pkl').exists():
            train_ids = np.load(LM_PATH / 'tmp' / 'train_ids.npy')
            val_ids = np.load(LM_PATH / 'tmp' / 'val_ids.npy')
            itos = pickle.load(open(LM_PATH / 'tmp' / 'itos.pkl', 'rb'))
        else:
            freq = collections.Counter(t for ts in tok_train for t in ts)
            itos = [t for t, k in freq.most_common(MAX_VOCAB) if k > MIN_FREQ]  # int idx to str token
            itos.insert(0, '_pad_')
            itos.insert(1, '_unk_')
            stoi = collections.defaultdict(lambda: 0, {t: i for i, t in enumerate(itos)})  # str token to int idx
            train_ids = np.array([[stoi[t] for t in ts] for ts in tok_train])
            val_ids = np.array([[stoi[t] for t in ts] for ts in tok_val])
            np.save(LM_PATH / 'tmp' / 'train_ids.npy', train_ids)
            np.save(LM_PATH / 'tmp' / 'val_ids.npy', val_ids)
            pickle.dump(itos, open(LM_PATH / 'tmp' / 'itos.pkl', 'wb'))

        vocab_size = len(itos)
        emb_dim, n_hidden, n_layers = 400, 1150, 3
        pre_path = PATH / 'models' / 'wt103'
        pre_lm_path = pre_path / 'fwd_wt103.h5'
        w = torch.load(pre_lm_path, map_location=lambda storage, loc: storage)
        enc_w = fastai.to_np(w['0.encoder.weight'])
        row_mean = enc_w.mean(0)
        itos_model = pickle.load((pre_path / 'itos_wt103.pkl').open('rb'))
        stoi_model = collections.defaultdict(lambda: -1, {t: i for i, t in enumerate(itos_model)})
        new_w = np.zeros((vocab_size, emb_dim), dtype=np.float32)
        for i, t in enumerate(itos):
            j = stoi_model[t]
            new_w[i] = enc_w[j] if j >= 0 else row_mean

        w['0.encoder.weight'] = fastai.T(new_w)
        w['0.encoder_with_dropout.embed.weight'] = fastai.T(np.copy(new_w))
        w['1.decoder.weight'] = fastai.T(np.copy(new_w))

        wd = 1e-7  # weight decay
        bptt = 70  # backpropagation through time, a.k.a. ngrams
        batch_size = 52
        optimizer_fn = functools.partial(torch.optim.Adam, betas=(0.8, 0.99))

        dl_train = ftext.LanguageModelLoader(np.concatenate(train_ids), batch_size, bptt)  # data loader
        dl_val = ftext.LanguageModelLoader(np.concatenate(val_ids), batch_size, bptt)
        md = ftext.LanguageModelData(PATH, 1, vocab_size, dl_train, dl_val, batch_size=batch_size, bptt=bptt)
        drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
        learner = md.get_model(optimizer_fn, emb_dim, n_hidden, n_layers,
                               dropouti=drops[0], dropout=drops[1], wdrop=drops[2],
                               dropoute=drops[3], dropouth=drops[4])
        learner.metrics = [fmetrics.accuracy]
        learner.freeze_to(-1)
        learner.model.load_state_dict(w)

        lr = 1e-3
        lrs = lr
        learner.fit(lrs / 2, 1, wds=wd, use_clr=(32, 2), cycle_len=1)
        learner.save('lm_last_ft')
        learner.lr_find(start_lr=lrs / 10, end_lr=lrs * 10, linear=True)
        learner.sched.plot()
        learner.fit(lrs, 1, wds=wd, use_clr=(20, 10), cycle_len=15)
        learner.save('lm1')
        learner.save_encoder('lm1_enc')
        learner.sched.plot_loss()

    if (CLAS_PATH / 'tmp' / 'tok_train.npy').exists():
        tok_train = np.load(CLAS_PATH / 'tmp' / 'tok_train.npy')
        tok_val = np.load(CLAS_PATH / 'tmp' / 'tok_val.npy')
        labels_train = np.load(CLAS_PATH / 'tmp' / 'labels_train.npy')
        labels_val = np.load(CLAS_PATH / 'tmp' / 'labels_val.npy')
    else:
        tok_train, labels_train = get_all(train_df, 1)
        tok_val, labels_val = get_all(val_df, 1)
        np.save(CLAS_PATH / 'tmp' / 'tok_train.npy', tok_train)
        np.save(CLAS_PATH / 'tmp' / 'tok_val.npy', tok_val)
        np.save(CLAS_PATH / 'tmp' / 'labels_train.npy', labels_train)
        np.save(CLAS_PATH / 'tmp' / 'labels_val.npy', labels_val)

    if (CLAS_PATH / 'tmp' / 'itos.pkl').exists():
        train_ids = np.load(CLAS_PATH / 'tmp' / 'train_ids.npy')
        val_ids = np.load(CLAS_PATH / 'tmp' / 'val_ids.npy')
        itos = pickle.load(open(CLAS_PATH / 'tmp' / 'itos.pkl', 'rb'))
    else:
        freq = collections.Counter(t for ts in tok_train for t in ts)
        itos = [t for t, k in freq.most_common(MAX_VOCAB) if k > MIN_FREQ]  # int idx to str token
        itos.insert(0, '_pad_')
        itos.insert(1, '_unk_')
        stoi = collections.defaultdict(lambda: 0, {t: i for i, t in enumerate(itos)})  # str token to int idx
        train_ids = np.array([[stoi[t] for t in ts] for ts in tok_train])
        val_ids = np.array([[stoi[t] for t in ts] for ts in tok_val])
        np.save(CLAS_PATH / 'tmp' / 'train_ids.npy', train_ids)
        np.save(CLAS_PATH / 'tmp' / 'val_ids.npy', val_ids)
        pickle.dump(itos, open(CLAS_PATH / 'tmp' / 'itos.pkl', 'wb'))

    vocab_size = len(itos)
    bptt = 70  # backpropagation through time, a.k.a. ngrams
    emb_dim, n_hidden, n_layers = 400, 1150, 3
    # optimizer_fn = functools.partial(optim.Adam, betas=(0.8, 0.99))
    batch_size = 48

    min_label = min(labels_train)
    labels_train -= min_label
    labels_val -= min_label
    k = int(max(labels_train)) + 1

    ds_train = ftext.TextDataset(train_ids, labels_train)
    ds_val = ftext.TextDataset(val_ids, labels_val)
    sampler_train = ftext.SortishSampler(train_ids, key=lambda x: len(train_ids[x]), bs=batch_size // 2)
    sampler_val = ftext.SortSampler(val_ids, key=lambda x: len(val_ids[x]))
    dl_train = dataloader.DataLoader(ds_train, batch_size // 2, transpose=True, num_workers=1, pad_idx=1,
                                     sampler=sampler_train)
    dl_val = dataloader.DataLoader(ds_val, batch_size // 2, transpose=True, num_workers=1, pad_idx=1,
                                   sampler=sampler_val)
    md = fdata.ModelData(PATH, dl_train, dl_val)

    # drops = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
    drops = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5
    model = lm_rnn.get_rnn_classifier(bptt, 20 * 70, k, vocab_size, emb_sz=emb_dim,
                                      n_hid=n_hidden, n_layers=n_layers,
                                      pad_token=1, layers=[emb_dim * 3, 50, k],
                                      drops=[drops[4], 0.1], dropouti=drops[0],
                                      wdrop=drops[1], dropoute=drops[2], dropouth=drops[3])
    optimizer_fn = functools.partial(torch.optim.Adam, betas=(0.7, 0.99))
    # learner = RNN_Learner(md, TextModel(to_gpu(model)), opt_fn=optimizer_fn)
    learner = ftext.RNN_Learner(md, ftext.TextModel(model), opt_fn=optimizer_fn)
    learner.reg_fn = functools.partial(lm_rnn.seq2seq_reg, alpha=2, beta=1)
    learner.clip = 25.0
    learner.metrics = [fmetrics.accuracy]

    # lr = 3e-3
    # lrm = 2.6
    # lrs = np.array([lr / lrm**4, lr / lrm**3, lr / lrm**2, lr / lrm, lr])
    lrs = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2])

    # wd = 1e-7  # weight decay
    wd = 0
    learner.load_encoder('lm1_enc')
    learner.freeze_to(-1)
    learner.lr_find(lrs / 1000)
    learner.sched.plot()
    learner.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))
    learner.save('clas_0')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Fastai Text Classifier')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--embeddings-size', dest='emb_dim', type=int, help='embeddings size')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='dimension of RNN hidden states')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--train-lang-model', dest='train_lm', help='train language model', action='store_true')
    parser.set_defaults(train_lm=False)
    args = parser.parse_args()

    run(vars(args))
