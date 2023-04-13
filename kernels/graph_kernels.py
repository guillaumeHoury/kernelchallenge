from kernels.base_kernels import Kernel
from utils.utils import cooccurrence_matrix

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs
import time
import multiprocessing as mp

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
    
    