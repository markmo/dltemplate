from argparse import ArgumentParser
from ml.doc_similarity.graph_features import GraphEdgeMaxCliqueSize, GraphPageRank
import pandas as pd


def run(constants):
    train_df = pd.read_pickle('../../entailment/notebooks/train_df.pkl')
    train_df = train_df[:20]
    # page_rank = GraphPageRank(train_df, constants,
    #                           weight_feature_name=None,
    #                           weight_feature_id=None,
    #                           reverse=False,
    #                           alpha=0.85,
    #                           max_iter=100)
    # df = page_rank.extract(train_df)
    edge_max_clique_size_feature = GraphEdgeMaxCliqueSize(train_df, constants)
    df = edge_max_clique_size_feature.extract(train_df)
    print(df.head())


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Doc Similarity model')
    parser.add_argument('--embed-type', dest='embedding_type', default='starspace', help='embedding type')
    args = parser.parse_args()

    run(vars(args))
