from kernels.base_kernels import Kernel, Linear

import numpy as np
import networkx as nx


class FeatureVectorKernel(Kernel):
    """
        Abstract class of kernels based on the vectorization of the graphs
    """

    def __init__(self, vector_kernel=Linear()):
        super().__init__()
        self.vector_kernel = vector_kernel

    def features(self, X):
        """
        :param X: list of graphs
        :return: vector of features
        """
        pass

    def kernel(self, X, Y):
        return self.vector_kernel.kernel(self.features(X), self.features(Y))


class NaiveKernel(FeatureVectorKernel):
    """
        A vectorizer kernel where features are the nb of nodes and edges of the graph.
    """
    def __init__(self):
        super().__init__()

    def features(self, X):
        X_ft = np.array([(G.number_of_nodes(), G.number_of_edges()) for G in X])
        return X_ft / np.mean(X_ft, axis=0)


class DegreeHistogramKernel(FeatureVectorKernel):
    """
    A vector kernel applied to the degree histogram of the graphs.
    """
    def __init__(self, max_degree=1):
        super().__init__()
        self.max_degree = max_degree

    def features(self, X):
        return np.array([np.bincount([min(d, self.max_degree) for n, d in G.degree()], minlength=self.max_degree+1) for G in X])


class NodeLabelHistogramKernel(FeatureVectorKernel):
    """
    A vector kernel applied to the degree histogram of the graphs.
    """
    def __init__(self, max_nb_labels=49):
        super().__init__()
        self.max_nb_labels = max_nb_labels

    def features(self, X):
        X_hist = np.array([np.bincount([G.nodes[n]['labels'][0] for n in G.nodes], minlength=self.max_nb_labels+1) for G in X])
        return X_hist / np.linalg.norm(X_hist, axis=1).reshape(-1, 1)


class EdgeLabelHistogramKernel(FeatureVectorKernel):
    """
    A vector kernel applied to the edge label count of the graphs.
    """
    def __init__(self, max_nb_labels=3):
        super().__init__()
        self.max_nb_labels = max_nb_labels

    def features(self, X):
        return np.array([np.bincount([G.edges[i, j]['labels'][0] for (i, j) in G.edges], minlength=self.max_nb_labels+1) for G in X])


class ShortestPathKernel(FeatureVectorKernel):
    """
    A vector kernel applied to the shortest path count vector of the graphs.
    """
    def __init__(self, max_length=1):
        super().__init__()
        self.max_length = max_length

    def features(self, X):
        X_shortest_path_lengths = [dict(nx.shortest_path_length(G)) for G in X]
        X_hist = np.array([np.bincount([min(d, self.max_length)
                                        for node in X_shortest_path_lengths[k].keys() for d in X_shortest_path_lengths[k][node].values()],
                                       minlength=self.max_length+1) for k in range(len(X))])
        return X_hist / np.linalg.norm(X_hist, axis=1).reshape(-1, 1)


class CycleKernel(FeatureVectorKernel):
    """
    A vector kernel applied to the cycle size vector of the graphs.
    """
    def __init__(self, max_length=1):
        super().__init__()
        self.max_length = max_length

    def features(self, X):
        X_cycles = np.array([np.bincount([min(len(cycle), self.max_length) for cycle in nx.cycle_basis(G)],
                                minlength=self.max_length + 1
                                ) for G in X])
        return X_cycles / (np.linalg.norm(X_cycles, axis=1).reshape(-1, 1) + 1e-6)
