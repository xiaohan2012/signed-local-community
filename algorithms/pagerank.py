import numpy as np
import networkx as nx
from scipy.sparse import diags, csr_matrix


def pr_score(g, q, alpha):
    """
    wrapper of pagerank score calculation 
    if graph is multi graph, use my own implementation of pagerank
    """
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        scores = pagerank_for_multi_graph(g, seeds=[q], alpha=alpha)
    else:
        personalization = {n: 0.0 for n in g.nodes()}
        personalization[q] = 1.0
        pr = nx.pagerank(g, alpha=1-alpha, personalization=personalization)
        scores = np.zeros(g.number_of_nodes())
        for v, s in pr.items():
            scores[v] = s
    return scores


def make_sparse_row_vector(n, positive_indices):
    cols = positive_indices
    rows = [0] * len(positive_indices)
    data = [1 / len(cols)] * len(positive_indices)

    return csr_matrix((data, (rows, cols)), shape=(1, n))


def pagerank_for_multi_graph(g, seeds=None, max_iter=100, tol=1.0e-6, alpha=0.15, verbose=0):
    """
    works for MultiGraph and MultiDiGraph because networkx does not support it
    """
    A = nx.adjacency_matrix(g)
    deg_inv = 1 / A.sum(axis=0)
    D_inv = diags(deg_inv.tolist()[0], 0)

    W = D_inv @ A

    s = make_sparse_row_vector(g.number_of_nodes(), seeds)
    p = make_sparse_row_vector(g.number_of_nodes(), list(range(g.number_of_nodes())))

    prev_p = p
    for i in range(max_iter):
        if verbose >= 2:
            print("iteration", i)
        p = (1 - alpha) * p @ W + alpha * s

        err = np.sum(np.absolute((prev_p - p).todense()))
        if err <= tol:
            break
        if verbose >= 2:
            print('error', err)
        prev_p = p
    return p.toarray().flatten()
