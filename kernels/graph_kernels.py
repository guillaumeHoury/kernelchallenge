from kernels.base_kernels import Kernel
from utils.utils import cooccurrence_matrix, optimized_coocurrence_matrix

import numpy as np

import networkx as nx
import multiprocessing as mp


class WLKernel(Kernel):
    """
    The Weisfeler-Lehman subraph kernel.
    """
    def __init__(self, iterations=3, distributed=True):
        super().__init__()
        self.iterations = iterations
        self.distributed = distributed

    def weisfeiler_lehman_subgraph_hashes(self, G):
        """
        Computes the WL subgraph hashes.
        :param G: A networkx graph.
        :return: A list of list of hashes (one list for each node).
        """
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

        K = np.array(cooccurrence_matrix(X_hash, Y_hash, distributed=self.distributed))

        return K

    def optimized_kernel(self, X, Y):
        """
        Optimized version of the kernel computation.
        """
        # Input lists of graphs X and Y of len N and M
        X_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in X]
        Y_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in Y]

        K = optimized_coocurrence_matrix(X_hash, Y_hash, distributed=self.distributed)

        return K


class NthWalkKernel(Kernel):
    """
    The Nth Walk kernel.
    """
    def __init__(self, edge_attr=False, node_attr=False, walk_length = 4):
        super().__init__()
        self.n = walk_length

    def mini_kernel(self, x):
        K_x = np.zeros(len(self.Y))
        for j, y in enumerate(self.Y):
            G_product = nx.tensor_product(x,y)
            " retrait des noeuds n'ayant pas les même label du graph produit"
            nodes_rem = []
            for nodes in G_product.nodes:
                a,b = (G_product.nodes[nodes]['labels'])
                if not (a==b): nodes_rem.append(nodes)
            for nodes in nodes_rem:
                G_product.remove_node(nodes)

            A = nx.to_numpy_array(G_product)
            A_n = np.linalg.matrix_power(A,self.n)
            K_x[j] = np.sum(A_n)
        return K_x

    def kernel(self, X, Y):
        pool = mp.Pool(10)
        self.Y = Y
        K_temp = pool.map(self.mini_kernel, X)
        K = np.array(K_temp)
        return K

