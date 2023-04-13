from kernels.base_kernels import Kernel
from utils.utils import cooccurrence_matrix, optimized_coocurrence_matrix

import numpy as np


class WLKernel(Kernel):
    """
    The Weisfeler-Lehman subraph kernel.
    """
    def __init__(self, iterations=3):
        super().__init__()
        self.iterations = iterations

    def weisfeiler_lehman_subgraph_hashes(self, G):
        # Initialize labels
        labels = {node: str(G.nodes[node]['labels']) for node in G.nodes()}

        final_hashes = {node: list() for node in G.nodes()}

        for _ in range(self.iterations):
            new_labels = {}
            for node in G.nodes():
                # Compute new features
                label_list = [str(G.edges[node,nbr]['labels']) + labels[nbr] for nbr in G.neighbors(node)]
                label = labels[node] + "".join(sorted(label_list))

                # Hash features
                hashed_label = str(hash(label))

                new_labels[node] = hashed_label
                final_hashes[node].append(hashed_label)

            labels = new_labels

        return final_hashes

    def kernel(self, X, Y):
        # Input lists of graphs X and Y of len N and M
        X_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in X]
        Y_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in Y]

        K = np.array(cooccurrence_matrix(X_hash, Y_hash))

        return K

    def optimized_kernel(self, X, Y, distributed=True):
        # Input lists of graphs X and Y of len N and M
        X_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in X]
        Y_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in Y]

        K = optimized_coocurrence_matrix(X_hash, Y_hash, distributed=distributed)

        return K