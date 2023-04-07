from kernels.base_kernels import Kernel
from utils.utils import cooccurrence_matrix

import numpy as np
import networkx as nx


class WLKernel(Kernel):
    """
    The Weisfeler-Lehman subraph kernel.
    """
    def __init__(self, edge_attr=False, node_attr=False, iterations=3, digest_size=16):
        super().__init__()
        self.edge_attr = 'labels' if edge_attr else None
        self.node_attr = 'labels' if node_attr else None
        self.iterations = iterations
        self.digest_size = digest_size

    def kernel(self, X, Y):
        # Input lists of graphs X and Y of len N and M
        X_hash = [sorted([subgraph_hash for vertex_hash in nx.weisfeiler_lehman_subgraph_hashes(G, edge_attr=self.edge_attr, node_attr=self.node_attr, iterations=self.iterations, digest_size=self.digest_size).values() for subgraph_hash in vertex_hash]) for G in X]
        Y_hash = [sorted(
            [subgraph_hash for vertex_hash in nx.weisfeiler_lehman_subgraph_hashes(G, edge_attr=self.edge_attr, node_attr=self.node_attr, iterations=self.iterations, digest_size=self.digest_size).values() for subgraph_hash in
             vertex_hash]) for G in Y]

        K = np.array(cooccurrence_matrix(X_hash, Y_hash))

        return K