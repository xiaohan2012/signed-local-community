import numpy as np
import networkx as nx
from scipy import sparse as sp
from tqdm import tqdm
from scipy.sparse import diags, issparse
from scipy.sparse.linalg import eigs, spsolve, cg
from numpy import linalg as LA

try:
    import cvxpy as cp
except:
    pass

from helpers import (
    signed_laplacian,
    signed_normalized_laplacian,
    prepare_seed_vector,
    prepare_seed_vector_sparse,
    degree_diag,
    is_rank_one,
    sbr_by_threshold,
    flatten,
    neg_adj, pos_adj
)


def query_graph(g, seeds, kappa=0.9, solver='cg', max_iter=30, tol=1e-3, verbose=0, return_details=False):
    """wrapper from different solvers"""
    assert solver in {'sp', 'sdp', 'cg'}
    assert kappa > 0, 'kappa should be non-negative'
    assert kappa < 1, 'kappa should be < 1'
    args = (g, seeds)
    kwargs = dict(kappa=kappa, verbose=verbose)
    if solver == 'sp':
        return query_graph_using_sparse_linear_solver(
            *args, **kwargs, solver='sp', max_iter=max_iter, return_details=return_details
        )
    elif solver == 'cg':
        n_attempts = 0
        while True and n_attempts <= 5:
            try:
                n_attempts += 1
                return query_graph_using_sparse_linear_solver(
                    *args, **kwargs, solver='cg', max_iter=max_iter, tol=tol, return_details=return_details
                )
            except RuntimeError:
                continue
        raise RuntimeError('cg error and attempted 5 times')

    elif solver == 'sdp':
        return query_graph_using_dense_matrix(*args, **kwargs)

    raise ValueError('unknown solver name', solver)


def query_graph_using_dense_matrix(g, seeds, kappa=0.9, verbose=0):
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


def query_graph_using_sparse_linear_solver(
        g, seeds,
        kappa=0.9,
        solver='cg',
        max_iter=40,
        tol=1e-3,
        ub=None,
        L=None,
        A=None,
        D=None,
        verbose=0,
        return_details=False
):
    """
    more scalable approach by solving a sparse linear system
    
    return:

    - x_opt: np.ndarray
    - opt_val: float
    """
    if solver == 'sp':
        print('WARNING: using "sp", note that "cg" is faster')

    if L is None:
        L = signed_laplacian(g)
    if A is None:
        A = nx.adj_matrix(g, weight='sign')
    if D is None:
        D = degree_diag(g)

    s = prepare_seed_vector_sparse(seeds, D)

    if verbose > 0:
        print('matrices loading done')

    lb = - g.number_of_edges() * 2
    if ub is None:
        Ln = signed_normalized_laplacian(A)
        lambda1 = eigs(Ln, k=1, which='SM')[0][0]  # the smallest eigen value
        ub = np.real(lambda1)
        if verbose > 0:
            print('found lambda_1=', lambda1)
    else:
        if verbose > 0:
            print('using given ub={}'.format(ub))
        assert ub >= 0, 'ub should be non-negative'

    b = D @ s
    n_steps = 0
    converged = False
    while n_steps <= max_iter:
        n_steps += 1
        alpha = np.real((ub + lb) / 2)
        A = L - alpha * D
        # linear system solver
        if solver == 'cg':
            if issparse(b):
                # cg requires b to be dense
                b = b.A
            y, info = cg(A, b)
            if info != 0:
                raise RuntimeError('cg error, info=', info)
        elif solver == 'sp':
            y = spsolve(A, b)
        else:
            raise ValueError('unknown solver', solver)
            
        y /= LA.norm(y, 2)
        y = diags(1 / np.sqrt(D.diagonal())) @ y[:, None]

        # assert np.isclose((y.T @ D @ y)[0, 0], 1), 'y not normalized w.r.t D'

        gap = np.real((np.sqrt(kappa) - y.T @ D @ s)[0, 0])

        if n_steps % 5 == 0 and verbose > 0:
            print('at iteration {} (alpha={:.5f})'.format(n_steps, alpha))
            print("residual: sqrt(kappa) - y' D s={}".format(gap))

        if gap > -tol and gap < 0:
            converged = True
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
    ret = flatten(y), y @ L @ y.T

    if return_details:
        details = dict(
            gap=gap,
            ub=ub,
            lb=lb,
            alpha=alpha,
            converged=converged,
            n_iters=n_steps,
            lambda1=lambda1
        )
        return ret + (details, )
    else:
        return ret


def sweep_on_x(g, x, top_k=-1, verbose=0):
    """
    g: the graph
    x: the node score vector
    top_k: the number of threshold to consider, -1 to consider all
    """
    ts = np.sort(np.abs(x))  # avoid very small threshold
    if top_k > 0:
        if verbose > 0:
            print('sweep on top {}'.format(top_k))
        ts = ts[-top_k:]  # use large absolute thresholds

    iters = (tqdm(ts) if verbose > 0 else ts)

    sbr_list = [sbr_by_threshold(g, x, t) for t in iters]
    best_t = ts[np.argmin(sbr_list)]
    best_sbr = np.min(sbr_list)
    if verbose > 0:
        print('best_t:', best_t)

    c1, c2 = np.nonzero(x <= -best_t)[0], np.nonzero(x >= best_t)[0]
    C = np.nonzero(np.abs(x) >= best_t)[0]

    if verbose > 0:
        print('comm1:', c1)
        print('comm2:', c2)
        
    return c1, c2, C, best_t, best_sbr, ts, sbr_list


def _add_by_order(seq1, seq2, order, x, verbose=0):
    """
    subroutine to be used by sweep_on_x_fast

    Given:
    
    seq1: values (can be pos/neg cut/internal-degree) ordered by x at each threshold
    seq2: values ordered by -x at each threshold
    order: node orders by |x|
    
    all parameters have equal length, n (number of nodes in graph)
    given some positive threshold t, denote the set of nodes above t as V1(t) and below -t as V2(t),
    we want to calcualte val1(V1(t)) + val2(V2(t)) at all thresholds t,
    where val1(V1(t)) means \sum_{v \in V1(t)} val1(v) and val2(V2(t)) means similarly
    
    Return:
    val1(V1(t)) + val2(V2(t)) at all thresholds t: length = n
    """
    res = np.ones(seq1.shape)
    nodes1, nodes2 = set(np.nonzero(x > 0)[0]), set(np.nonzero(x < 0)[0])
    i, j, k = -1, -1, -1
    for n in order:
        if n in nodes1:
            i += 1
            if verbose > 0:
                print('-> set 1: i=', i)
        if n in nodes2:
            j += 1
            if verbose > 0:
                print('-> set 2: j=', j)
        k += 1
        if verbose > 0:
            print('current position: ', k)
        val1 = (seq1[i] if i >= 0 else 0)
        val2 = (seq2[j] if j >= 0 else 0)
        res[k] = val1 + val2
        if verbose > 0:
            print('res[{}]={}'.format(k, res[k]))
    return res


def sweep_on_x_fast(g, x, top_k=-1, A=None, return_details=False, verbose=0):
    """
    sweep on x in one go (no iteration on thresholds)
    
    returns:
        C1, C2, C, best_t, best_beta, ts, beta_array
    """
    if A is None:
        A = nx.adj_matrix(g, weight='sign')

    pos_A, neg_A = pos_adj(A), neg_adj(A)

    # calculate volume
    abs_order = np.argsort(np.abs(x))[::-1]
    # print('abs_order', abs_order)
    pos_A_by_abs = pos_A[abs_order, :][:, abs_order]
    neg_A_by_abs = neg_A[abs_order, :][:, abs_order]

    pos_vol_by_abs = np.cumsum(flatten(pos_A_by_abs.sum(axis=1)))
    neg_vol_by_abs = np.cumsum(flatten(neg_A_by_abs.sum(axis=1)))
    vol_by_abs = pos_vol_by_abs + neg_vol_by_abs

    # print('pos_vol_by_abs', pos_vol_by_abs)
    # print('neg_vol_by_abs', neg_vol_by_abs)
    # print('vol_by_abs', vol_by_abs)

    ##### calculate the cut by abs(x)
    # = cut(V1 \cup V2, complement of V1 \cup V2)

    pos_A_by_abs_lower = sp.tril(pos_A_by_abs)
    neg_A_by_abs_lower = sp.tril(neg_A_by_abs)
    pos_cut_by_abs = pos_vol_by_abs - np.cumsum(2*flatten(pos_A_by_abs_lower.sum(axis=1)))
    neg_cut_by_abs = neg_vol_by_abs - np.cumsum(2*flatten(neg_A_by_abs_lower.sum(axis=1)))
    cut_by_abs = pos_cut_by_abs + neg_cut_by_abs
    # print('abs_order', abs_order)
    # print('pos_cut_by_abs', pos_cut_by_abs)
    # print('neg_cut_by_abs', neg_cut_by_abs)
    # print('cut_by_abs', cut_by_abs)
    # print('neg_cut', neg_cut)

    ##### calculate negative/positive degree **inside**
    # node order by x from high to low
    pos_order = np.argsort(np.maximum(x, 0))[::-1]

    # node order by -x from high to low
    neg_order = np.argsort(np.maximum(-x, 0))[::-1]
    # print('pos order', pos_order)
    # print('neg order', neg_order)

    # pos nodes on top
    neg_A_1 = neg_A[pos_order, :][:, pos_order]
    neg_A_lower_1 = sp.tril(neg_A_1)
    neg_at_k_1 = flatten(2 * neg_A_lower_1.sum(axis=1))
    neg_inside_1 = np.cumsum(neg_at_k_1)

    # neg nodes on top
    neg_A_2 = neg_A[neg_order, :][:, neg_order]
    neg_A_lower_2 = sp.tril(neg_A_2)
    neg_at_k_2 = flatten(2 * neg_A_lower_2.sum(axis=1))
    neg_inside_2 = np.cumsum(neg_at_k_2)

    # print('neg_inside_1', neg_inside_1)
    # print('neg_inside_2', neg_inside_2)
    
    pos_A_1 = pos_A[pos_order, :][:, pos_order]
    pos_A_lower_1 = sp.tril(pos_A_1)
    pos_at_k_1 = flatten(2 * pos_A_lower_1.sum(axis=1))
    pos_inside_1 = np.cumsum(pos_at_k_1)

    pos_A_2 = pos_A[neg_order, :][:, neg_order]
    pos_A_lower_2 = sp.tril(pos_A_2)
    pos_at_k_2 = flatten(2 * pos_A_lower_2.sum(axis=1))
    pos_inside_2 = np.cumsum(pos_at_k_2)

    # print('pos order', pos_order)
    # print('neg order', neg_order)

    # print('pos_at_k_1', flatten(pos_at_k_1))
    # print('pos_at_k_2', flatten(pos_at_k_2))
    # print('pos_inside_1', pos_inside_1)
    # print('pos_inside_2', pos_inside_2)

    ######## positive cut on V1 and V2
    pos_vol_1 = np.cumsum(flatten(pos_A_1.sum(axis=1)))
    pos_vol_2 = np.cumsum(flatten(pos_A_2.sum(axis=1)))

    pos_cut_1 = pos_vol_1 - pos_inside_1
    pos_cut_2 = pos_vol_2 - pos_inside_2
    # print('pos_cut_1', pos_cut_1)
    # print('pos_cut_2', pos_cut_2)

    ####### the elixir #######
    ####### positive V1 and V2 in between #######
    pos_cut_1_and_2 = _add_by_order(pos_cut_1, pos_cut_2, abs_order, x)
    pos_between_1_2 = pos_cut_1_and_2 - pos_cut_by_abs

    # count the negative inside V1 and V2 separately and then add up
    neg_inside_1_2 = _add_by_order(neg_inside_1, neg_inside_2, abs_order, x)

    beta_array = (pos_between_1_2 + neg_inside_1_2 + cut_by_abs) / vol_by_abs
    # print('beta:', beta_array)
    if top_k == -1:
        top_k = beta_array.shape[0]

    assert top_k >= 1
    
    beta_array = beta_array[:top_k]

    best_idx = np.argmin(beta_array)
    best_beta = np.min(beta_array)
    # print('best position: ', best_idx+1)
    C = abs_order[:best_idx+1]
    C1 = C[x[C] > 0]
    C2 = C[x[C] < 0]

    ts = np.sort(np.abs(x))[::-1][:top_k]
    best_t = ts[best_idx]

    ret = (C1, C2, C, best_t, best_beta, ts, beta_array)

    if verbose > 0:
        print('pos_order', pos_order)
        print('neg_order', neg_order)
        print('abs_order', abs_order)
        print('pos_vol_by_abs', pos_vol_by_abs)
        print('neg_vol_by_abs', neg_vol_by_abs)
        print('vol_by_abs', vol_by_abs)
        print('pos_cut_by_abs', pos_cut_by_abs)
        print('neg_cut_by_abs', neg_cut_by_abs)
        print('neg_inside_1', neg_inside_1)
        print('neg_inside_2', neg_inside_2)
        print('pos_inside_1', pos_inside_1)
        print('pos_inside_2', pos_inside_2)
        print('pos_cut_1', pos_cut_1)
        print('pos_cut_2', pos_cut_2)
        print('pos_between_1_2', pos_between_1_2)
        print('neg_inside_1_2', neg_inside_1_2)
    if return_details:
        details = dict(
            pos_A=pos_A,
            neg_A=neg_A,
            pos_order=pos_order,
            neg_order=neg_order,
            abs_order=abs_order,
            pos_vol_by_abs=pos_vol_by_abs,
            neg_vol_by_abs=neg_vol_by_abs,
            pos_cut_by_abs=pos_cut_by_abs,
            neg_cut_by_abs=neg_cut_by_abs,
            neg_inside_1=neg_inside_1,
            neg_inside_2=neg_inside_2,
            pos_inside_1=pos_inside_1,
            pos_inside_2=pos_inside_2,
            pos_cut_1=pos_cut_1,
            pos_cut_2=pos_cut_2,
            pos_between_1_2=pos_between_1_2,
            neg_inside_1_2=neg_inside_1_2
        )
        return ret + (details, )
    else:
        return ret
