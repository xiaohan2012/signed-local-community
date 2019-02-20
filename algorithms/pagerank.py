import numpy as np
import networkx as nx
from scipy.sparse import diags, csr_matrix


def _pagerank_scipy(G, alpha=0.85, personalization=None,
                    max_iter=100, tol=1.0e-6, weight='weight',
                    dangling=None):
    """
    copied from networkx, but removes the `raise` part
    """
    
    import scipy.sparse

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p = p / p.sum()

    # intial vector
    x = p.copy()
    
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    # return what I have, though not convergent
    return dict(zip(nodelist, map(float, x)))


def pr_score(g, q, alpha, max_iter=100):
    """
    wrapper of pagerank score calculation
    if graph is multi graph, use my own implementation of pagerank
    """
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        scores = pagerank_for_multi_graph(
            g,
            seeds=[q],
            alpha=alpha,
            max_iter=max_iter
        )
    else:
        personalization = {n: 0.0 for n in g.nodes()}
        personalization[q] = 1.0
        pr = _pagerank_scipy(
            g,
            alpha=1-alpha,
            personalization=personalization,
            max_iter=max_iter
        )
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
