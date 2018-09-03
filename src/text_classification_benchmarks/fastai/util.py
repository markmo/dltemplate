from common.load_data import DATA_DIR
from fastai.text import *
import html
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# noinspection SpellCheckingInspection
BOS, FLD = 'xbos', 'xfld'


def clean_text(text):
    # remove non-latin chars
    text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
    # remove whitespace at ends
    text = text.strip()
    # replace whitespace with single spaces
    text = re.sub(r'\s+', ' ', text)
    # remove non-printable chars
    return ''.join(ch for ch in text if ch.isprintable())


# noinspection SpellCheckingInspection
def fixup(text):
    replacements = [('#39;', "'"), ('amp;', '&'), ('#146;', "'"), ('nbsp;', ' '), ('#36;', '$'),
                    ('\\n', '\n'), ('quot;', "'"), ('<br />', '\n'), ('<br/>', '\n'), ('<br>', '\n'),
                    ('\\"', '"'), ('<unk>', 'u_n'), (' @.@ ', '.'), (' @-@ ', '-'), ('\\', ' \\ ')]
    for p, r in replacements:
        text = text.replace(p, r)

    p = re.compile(r'( )\1{2,}')
    return p.sub(' ', html.unescape(text))


def get_all(df, n_labels):
    tok, labels = [], []
    for i, r in enumerate(df):
        tok_, labels_ = get_texts(r, n_labels)
        tok += tok_
        labels += labels_

    return tok, labels


def get_texts(df, n_labels=1):
    labels = df.iloc[:, range(n_labels)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_labels].astype(str)
    for i in range(n_labels + 1, len(df.columns)):
        texts += f' {FLD} {i - n_labels} ' + df[i].astype(str)

    texts = list(texts.apply(fixup).values)
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def preprocess_csv():
    """ Convert exported CSV file from Watson to fasttext format. """
    filename = DATA_DIR + 'text_classification/codi/intents.csv'
    df = pd.read_csv(filename, header=None)
    df = df.dropna()
    classes = df[1].unique()
    class_list = classes.tolist()
    df[0] = df[0].apply(clean_text)
    df[1] = df[1].apply(lambda x: class_list.index(x))
    counts = df[1].value_counts()

    # omit classes with too few examples
    omit = counts[counts < 2].index.values
    omitted = df[df[1].isin(omit)]
    included = df[~df[1].isin(omit)]
    y = included.pop(1)

    x_train, x_test, y_train, y_test = train_test_split(included, y, test_size=0.1, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train,
                                                      random_state=42)
    train_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)
    val_df: pd.DataFrame = pd.concat([y_val, x_val], axis=1)
    test_df: pd.DataFrame = pd.concat([y_test, x_test], axis=1)

    # add omitted examples back to training sets
    train_df: pd.DataFrame = pd.concat([train_df, omitted], axis=0)
    train_df = train_df.reindex(columns=[1, 0])
    x_train: pd.DataFrame = pd.concat([x_train, omitted[0]], axis=0)
    y_train: pd.DataFrame = pd.concat([y_train, omitted[1]], axis=0)

    # save to file
    train_df.to_csv('train.csv', header=False, index=False)
    val_df.to_csv('val.csv', header=False, index=False)
    test_df.to_csv('test.csv', header=False, index=False)
    np.savetxt('classes.txt', classes, fmt='%s')

    return (train_df, val_df, test_df,
            x_train.values, y_train.values, x_val.values, y_val.values, x_test.values, y_test.values, classes)
