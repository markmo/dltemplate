from common.load_data import DATA_DIR
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split


# noinspection SpellCheckingInspection
def clean_text(text):
    # remove non-latin chars (np.savetxt supports only latin-1)
    text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
    # remove whitespace at ends
    text = text.strip()
    # replace whitespace with single spaces
    text = re.sub(r'\s+', ' ', text)
    # remove non-printable chars
    return ''.join(ch for ch in text if ch.isprintable())


def convert_label(label):
    """ Conform to expected fasttext label format. """
    return '__label__{}'.format(label)


# noinspection SpellCheckingInspection,PyTypeChecker
def preprocess_csv():
    """ Convert exported CSV file from Watson to fasttext format. """
    filename = DATA_DIR + 'text_classification/codi/intents.csv'
    df = pd.read_csv(filename, header=None)
    df = df.dropna()
    df[0] = df[0].apply(clean_text)
    df[1] = df[1].apply(convert_label)
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
    val_df: pd.DataFrame = pd.concat([x_val, y_val], axis=1)
    test_df: pd.DataFrame = pd.concat([x_test, y_test], axis=1)

    # add omitted examples back to training sets
    train_df: pd.DataFrame = pd.concat([train_df, omitted], axis=0)
    x_train: pd.DataFrame = pd.concat([x_train, omitted[0]], axis=0)
    y_train: pd.DataFrame = pd.concat([y_train, omitted[1]], axis=0)

    # save to file
    # df.to_csv('data.csv', header=False, index=False)
    # np.savetxt('data.txt', df.values, fmt='%s')
    np.savetxt('train.txt', train_df.values, fmt='%s')
    np.savetxt('val.txt', val_df.values, fmt='%s')
    np.savetxt('test.txt', test_df.values, fmt='%s')

    return x_train.values, y_train.values, x_val.values, y_val.values, x_test.values, y_test.values
