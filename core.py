import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigs, spsolve, cg
from numpy import linalg as LA
from scipy.sparse import diags

import cvxpy as cp

from helpers import (
    signed_laplacian,
    prepare_seed_vector,
    degree_diag,
    is_rank_one,
    sbr_by_threshold,
    flatten
)


def query_graph(g, seeds, kappa=0.25, solver='sp', verbose=0):
    """wrapper from different solvers"""
    assert solver in {'sp', 'sdp'}
    args = (g, seeds)
    kwargs = dict(kappa=kappa, verbose=verbose)
    if solver == 'sp':
        return query_graph_using_sparse_linear_solver(*args, **kwargs)
    elif solver == 'sdp':
        return query_graph_using_dense_matrix(*args, **kwargs)

    raise ValueError('unknown solver name', solver)


def query_graph_using_dense_matrix(g, seeds, kappa=0.25, verbose=0):
    n = g.number_of_nodes()
    L = signed_laplacian(g).A
    D = degree_diag(g).A

    s = prepare_seed_vector(seeds, D)

    # requirement check
    _, v1 = eigs(np.real(L), k=1, which='SM')
    assert not np.isclose((s.T @ np.real(v1))[0, 0], 0)

    DsDsT = (D @ s) @ (D @ s).T

    X = cp.Variable((n, n))
    constraints = [
        X >> 0,
        cp.trace(D @ X) == 1,
        cp.trace(DsDsT @ X) >= kappa
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(L @ X)), constraints)
    opt_val = prob.solve()
    
    if verbose > 0:
        print("is rank one? ", is_rank_one(X.value, False))
    
    x_opt = X.value[0, :]

    return x_opt, opt_val


def query_graph_using_sparse_linear_solver(g, seeds, kappa=0.25, tol=1e-3, verbose=0):
    """
    more scalable approach by solving a sparse linear system
    """
    L = signed_laplacian(g)
    D = degree_diag(g)

    s = prepare_seed_vector(seeds, D)
    
    lb = - g.number_of_edges() * 2
    lambda1 = eigs(L, k=1, which='SM')[0][0]  # the smallest eigen value
    ub = np.real(lambda1)

    b = D @ s
    n_steps = 0
    while True:
        n_steps += 1
        alpha = (ub + lb) / 2
        A = L - alpha * D
        # linear system solver
        y = spsolve(A, b)
            
        y /= LA.norm(y, 2)
        y = diags(1 / np.sqrt(D.diagonal())) @ y[:, None]

        assert np.isclose((y.T @ D @ y)[0, 0], 1), 'y not normalized w.r.t D'

        gap = (np.sqrt(kappa) - y.T @ D @ s)[0, 0]
        
        if gap > -tol and gap < 0:
            if verbose > 0:
                print("""terminates after {} iterations:
  - alpha={:.5f}
  - residual={:.5f}""".format(n_steps, alpha, gap))
            break

        if gap < 0:
            lb = alpha
        else:
            ub = alpha
    y = y.T  # make it row
    return flatten(y), y @ L @ y.T


def sweep_on_x(g, x, verbose=0):
    """
    g: the graph
    x: the node score vector
    """
    ts = sorted(np.abs(x))[2:]  # avoid very small threshold
    sbr_list = [sbr_by_threshold(g, x, t) for t in ts]
    best_t = ts[np.argmin(sbr_list)]
    best_sbr = np.min(sbr_list)
    if verbose > 0:
        print('best_t:', best_t)

    c1, c2 = np.nonzero(x <= -best_t)[0], np.nonzero(x >= best_t)[0]
    C = np.nonzero(np.abs(x) >= best_t)[0]

    if verbose > 0:
        print('comm1:', c1)
        print('comm2:', c2)
        
    return c1, c2, C, best_sbr, ts, sbr_list
