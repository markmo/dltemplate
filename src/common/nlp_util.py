import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tf_idf_matrix(texts):
    # `min_df`, minimum document frequency - cut-off threshold for low frequency n-grams
    # `max_df`, maximum document frequency - cut-off threshold for high frequency n-grams such as stop words
    # `ngram_range` - which n-grams should be used in this bag-of-words (BOW) representation
    tf_idf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    features = tf_idf.fit_transform(texts)
    return pd.DataFrame(features.todense(), columns=tf_idf.get_feature_names())
