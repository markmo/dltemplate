from collections import defaultdict
import logging
from ml.doc_similarity.extractor import Extractor
from ml.doc_similarity.math_util import aggregate
import networkx as nx
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class Graph(Extractor):

    @staticmethod
    def generate_graph_clique(graph):
        n2clique = defaultdict(list)  # node (qid) to clique dict
        cliques = []
        for clique in nx.find_cliques(graph):
            for n in clique:
                n2clique[n].append(len(cliques))

            cliques.append(clique)

        logging.info('Initialized graph cliques, length: %d' % len(cliques))
        return n2clique, cliques

    @staticmethod
    def generate_graph_cc(graph):
        n2cc = {}  # node (qid) to connected-component dict
        ccs = []
        for cc in nx.connected_components(graph):
            for n in cc:
                if n in n2cc:
                    logging.warning('%d already in n2cc (%d)' % (n, n2cc[n]))

                n2cc[n] = len(ccs)

            ccs.append(cc)

        logging.info('Initialized graph cc, length: %d' % len(ccs))
        return n2cc, ccs

    @staticmethod
    def generate_pagerank(graph, alpha, max_iter):
        rank = nx.pagerank(graph, alpha=alpha, max_iter=max_iter)
        logging.info('Graph cal pagerank done')
        return rank

    @staticmethod
    def generate_hits(graph, max_iter):
        # returns dicts keyed by node containing the hub and authority values
        hits_h, hits_a = nx.hits(graph, max_iter=max_iter)
        logging.info('Graph cal hits done')
        return hits_h, hits_a


class DirectedGraph(Graph):
    pass


class UndirectedGraph(Graph):

    def __init__(self, df, config, weight_feature_name=None, weight_feature_id=None, reverse=False):
        super().__init__(config)
        if weight_feature_name is None and weight_feature_id is None and not reverse:
            self.feature_name = self.__class__.__name__
        else:
            self.feature_name = \
                '{}_{}_{}_{}'.format(self.__class__.__name__, weight_feature_name, weight_feature_id, reverse)

        self.edge2weight, self.graph = \
            UndirectedGraph.generate_graph(df, config, weight_feature_name, weight_feature_id, reverse)

    @staticmethod
    def generate_graph(df, config, weight_feature_name, weight_feature_id, reverse):
        edge2weight = {}
        graph = nx.Graph()

        # train_wfs_fs = None
        # test_wfs_fs = None

        i = 0
        for qid1, qid2 in zip(df.qid1.values, df.qid2.values):
            weight = 0
            graph.add_edge(qid1, qid2, weight=weight)
            edge2weight[(qid1, qid2)] = weight
            edge2weight[(qid2, qid1)] = weight
            i += 1

        logging.info('Graph constructed')
        return edge2weight, graph


class GraphEdgeMaxCliqueSize(UndirectedGraph):
    """Max clique size of the edge"""

    def __init__(self, df, config):
        super().__init__(df, config)
        # extract clique from graph
        self.n2clique, self.cliques = Graph.generate_graph_clique(self.graph)

    def extract_row(self, row):
        edge_max_clique_size = 0
        for clique_id in self.n2clique[row['qid1']]:
            if row['qid2'] in self.cliques[clique_id]:
                edge_max_clique_size = max(edge_max_clique_size, len(self.cliques[clique_id]))

        return [edge_max_clique_size]


class GraphNodeMaxCliqueSize(GraphEdgeMaxCliqueSize):

    def extract_row(self, row):
        lnode_max_clique_size = 0
        rnode_max_clique_size = 0
        for clique_id in self.n2clique[row['qid1']]:
            lnode_max_clique_size = max(lnode_max_clique_size, len(self.cliques[clique_id]))

        for clique_id in self.n2clique[row['qid2']]:
            rnode_max_clique_size = max(rnode_max_clique_size, len(self.cliques[clique_id]))

        return [lnode_max_clique_size,
                rnode_max_clique_size,
                max(lnode_max_clique_size, rnode_max_clique_size),
                min(lnode_max_clique_size, rnode_max_clique_size)]


class GraphNumClique(UndirectedGraph):

    def __init__(self, df, config):
        super().__init__(df, config)
        # extract clique from graph
        self.n2clique, self.cliques = Graph.generate_graph_clique(self.graph)

    def extract_row(self, row):
        n_cliques = 0
        for clique_id in self.n2clique[row['qid1']]:
            if row['qid2'] in self.cliques[clique_id]:
                n_cliques += 1

        return [n_cliques]


class GraphEdgeCCSize(UndirectedGraph):

    def __init__(self, df, config):
        super().__init__(df, config)
        self.n2cc, self.ccs = Graph.generate_graph_cc(self.graph)

    def extract_row(self, qid1):
        edge_cc_size = len(self.ccs[self.n2cc[qid1]])
        return [edge_cc_size]


class GraphPageRank(UndirectedGraph):

    def __init__(self, df, config, weight_feature_name, weight_feature_id, reverse, alpha, max_iter):
        super().__init__(df, config, weight_feature_name, weight_feature_id, reverse)
        self.feature_name = '{}_{}_{}'.format(self.feature_name, alpha, max_iter)
        self.pr = Graph.generate_pagerank(self.graph, alpha, max_iter)

    def extract_row(self, row):
        pr1 = self.pr[row['qid1']] * 1e6
        pr2 = self.pr[row['qid2']] * 1e6
        return [pr1, pr2, max(pr1, pr2), min(pr1, pr2), (pr1 + pr2) / 2.]


class GraphHits(UndirectedGraph):

    def __init__(self, df, config, weight_feature_name=None, weight_feature_id=None,
                 reverse=False, max_iter=100):
        super().__init__(df, config, weight_feature_name, weight_feature_id, reverse)
        self.hits_h, self.hits_a = Graph.generate_hits(self.graph, max_iter)

    def extract_row(self, row):
        qid1 = row['qid1']
        qid2 = row['qid2']
        h1 = self.hits_h[qid1] * 1e6
        h2 = self.hits_h[qid2] * 1e6
        a1 = self.hits_a[qid1] * 1e6
        a2 = self.hits_a[qid2] * 1e6
        return [h1, h2, a1, a2,
                max(h1, h2), max(a1, a2),
                min(h1, h2), min(a1, a2),
                (h1 + h1) / 2., (a1 + a2) / 2.]


class GraphShortestPath(UndirectedGraph):

    def extract_row(self, row):
        qid1 = row['qid1']
        qid2 = row['qid2']
        shortest_path = -1
        self.graph.remove_edge(qid1, qid2)
        if nx.has_path(self.graph, qid1, qid2):
            shortest_path = nx.dijkstra_path_length(self.graph, qid1, qid2)

        self.graph.add_edge(qid1, qid2, weight=self.edge2weight[(qid1, qid2)])
        return [shortest_path]


class GraphNodeNeighborProperty(UndirectedGraph):

    def extract_row(self, row):
        qid1 = row['qid1']
        qid2 = row['qid2']
        l, r = [], []
        l_nb = self.graph.neighbors(qid1)
        r_nb = self.graph.neighbors(qid2)
        for n in l_nb:
            if (n != qid2) and (n != qid1):
                l.append(self.edge2weight[(qid1, n)])

        for n in r_nb:
            if (n != qid2) and (n != qid1):
                r.append(self.edge2weight[(qid2, n)])

        agg_modes = ['mean', 'std', 'max', 'min', 'median']
        fs = aggregate(l, agg_modes) + aggregate(r, agg_modes)
        return fs


class GraphNodeNeighborShareNum(UndirectedGraph):

    def extract_row(self, row):
        l_nb = self.graph.neighbors(row['qid1'])
        r_nb = self.graph.neighbors(row['qid2'])
        return [len(list((set(l_nb).union(set(r_nb))) ^ (set(l_nb) ^ set(r_nb))))]
