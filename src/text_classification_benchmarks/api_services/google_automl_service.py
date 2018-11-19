import os


def create_dataset(train_df, filename='code_automl.csv'):
    train_df[['utterance', 'label']].to_csv(filename, header=False, index=False)
    return os.path.abspath(filename)
