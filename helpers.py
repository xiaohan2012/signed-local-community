import numpy as np
import random
import string
import datetime
import hashlib
import scipy
import sys

import pandas as pd
import networkx as nx

from collections import defaultdict
from scipy.sparse import diags, coo_matrix
from scipy.sparse.linalg import eigs, norm as norms
from scipy import sparse as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.linalg import sqrtm, inv
from numpy.linalg import svd


def walk(g, s0, beta, n_steps, verbose=0):
    """
    when the walker travels a "-" edge, it simply jumps back to s0

    with beta probability to jump back to s0
    """
    counter = np.zeros(g.number_of_nodes())
    current = s0
    for _ in range(n_steps):
        counter[current] += 1
        if verbose >= 2:
            print('current: ', current)
        r = random.random()
        if r <= beta:
            # teleport to s0
            next_nbr = s0
        else:
            nbrs = list(g.neighbors(current))
            next_nbr = random.choice(nbrs)
            sign = g[current][next_nbr]['sign']
            if sign == -1:
                next_nbr = s0
        if verbose >= 2:
            print('edge sign: ', sign)
        current = next_nbr

    return counter / n_steps


def signed_conductance(g, S, verbose=0):
    """
    fraction of edges from nodes S that are either + outside or - inside
    
    |cut+(S, V\S) + #deg-(S)| / sum_s deg(s)
    """
    raise Exception('this implementation is wrong!')
    numer = 0
    denum = 0
    S = set(S)
    for u in S:
        for v in g.neighbors(u):
            if v in S and g[u][v]['sign'] == -1:
                numer += 1
            elif v not in S and g[u][v]['sign'] == 1:
                numer += 1
            denum += 1
    if verbose >= 1:
        print('{} / {}'.format(numer, denum))
    denum = min(denum, 2 * g.number_of_edges() - denum)
    return numer / denum


def signed_laplacian(g):
    deg = nx.adjacency_matrix(g).sum(axis=0)
    D = diags(deg.tolist()[0], 0)
    A = nx.adjacency_matrix(g, weight='sign')
    L = D - A
    return L


def signed_layout(g, normalize=False):
    if normalize:
        L = signed_normalized_laplacian(nx.adj_matrix(g, weight='sign'))
    else:
        L = signed_laplacian(g)

    w, pos_array = eigs(L.asfptype(), k=2, which='SM')
    pos_array = np.real(pos_array)
    return {i: pos_array[i, :] for i in range(g.number_of_nodes())}


def pos_spring_layout(g, normalize=False):
    """spring layout using the positive graph"""
    A = nx.adj_matrix(g, weight='sign')
    pos_A = pos_adj(A)
    new_g = nx.from_scipy_sparse_matrix(pos_A)
    return nx.spring_layout(new_g)


def signed_layout_geometric_mean(g):
    """
    visualization using geometric mean of Laplacian
    Reference:
    Clustering Signed Networks with the Geometric Mean of Laplacians, NIPS 2016
    note: supporting only **small** graphs (internally using numpy.array)
    """
    
    A = nx.adjacency_matrix(g, weight='sign')
    Ap = pos_adj(A).A
    An = neg_adj(A).A
    Lp = normalized_laplacian(Ap).A
    Qn = normalized_laplacian(An, subtract=False).A

    Lp_sq = sqrtm(Lp)
    Lp_sq_inv = inv(Lp_sq)
    L_gm = Lp_sq @ sqrtm(Lp_sq_inv @ Qn @ Lp_sq_inv) @ Lp_sq

    w, pos_array = eigs(np.real(L_gm), k=2, which='SM')
    pos_array = np.real(pos_array)
    pos = {i: pos_array[i, :] for i in range(g.number_of_nodes())}
    return pos


def draw_nodes(g, pos, labels=None, ax=None):
    nx.draw_networkx_nodes(g, pos, ax=ax)
    if labels:
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax)


def draw_edges(g, pos, ax=None, draw_pos=True, draw_neg=True, **kwargs):
    if draw_neg:
        neg_edges = [(u, v) for u, v in g.edges() if g[u][v]['sign'] < 0]
        nx.draw_networkx_edges(g, pos, neg_edges, style='dashed', edge_color='red', ax=ax, **kwargs)
    
    if draw_pos:
        pos_edges = [(u, v) for u, v in g.edges() if g[u][v]['sign'] > 0]
        nx.draw_networkx_edges(g, pos, pos_edges, style='solid', edge_color='blue', ax=ax, **kwargs)


def show_result(g, pos, query, scores):
    normalized_scores = scores / degree_array(g)
    order = np.argsort(normalized_scores)[::-1]
    sorted_scores = scores[order]

    print('nodes sorted by PR score', order)
    print('PR scores after sorting', sorted_scores)

    fig, ax = plt.subplots(1, 1)
    nx.draw_networkx_nodes(
        g, pos, node_color=np.log2((normalized_scores + 1e-5) * 1e5), cmap='Blues')
    nx.draw_networkx_labels(g, pos)
    draw_edges(g, pos)
    ax.set_title('query node {}'.format(query))

    # sweeping plot
    sweep_positions = []
    sweep_scores = []
    for i in range(1, len(order)+1):
        if normalized_scores[order[i-1]] == 0:
            break
        sweep_positions.append(i)
        s = signed_conductance(g, order[:i])
        sweep_scores.append(s)

    fig, ax = plt.subplots(1, 1)
    ax.plot(sweep_positions, sweep_scores)
    ax.set_title('query node {}'.format(query))
    ax.set_xlabel('sweeping position')
    ax.set_ylabel('signed conductance')
    
    # get the best community
    best_pos = np.argmin(sweep_scores)
    comm = order[:best_pos+1]
    print('best position', best_pos)
    print('community', comm)
    
    fig, ax = plt.subplots(1, 1)
    color = np.zeros(g.number_of_nodes())
    color[comm] = 1
    nx.draw_networkx_nodes(g, pos, node_color=color, cmap='Blues')
    nx.draw_networkx_labels(g, pos)
    draw_edges(g, pos)
    ax.set_title('query node {}'.format(query))


def evaluate_performance(g, pred_comm, true_comm):
    prec = len(pred_comm.intersection(true_comm)) / len(true_comm)
    recall = len(pred_comm.intersection(true_comm)) / len(true_comm)
    f1 = 2 * prec * recall / (prec + recall)

    c = signed_conductance(g, pred_comm)

    return dict(prec=prec, recall=recall, f1=f1, conductance=c)


def get_now():
    return datetime.date.today().strftime("%Y-%m-%d %H:%M:%s")


def make_range(start, end, step=0.1):
    return np.arange(start, end + 0.1 * step, step)


def random_str(N=8):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return round(n * multiplier) / multiplier


def degree_array(g):
    return np.array(
        [g.degree[v]
         for v in np.arange(g.number_of_nodes())]
    )


def make_pair(u, v):
    return tuple(sorted([u, v]))


def _one_step_for_incremental_signed_conductance(
        g, prev_nodes, new_node, prev_vol, prev_pos_cut, prev_neg_cut, verbose=2
):
    raise Exception('deprecated, do not use')
    if verbose > 1:
        print('prev_nodes', prev_nodes)
        print('new_node', new_node)
        
    new_vol = prev_vol + g.degree(new_node)
    pos_cut_to_remove = {make_pair(u, new_node)
                         for u in g.neighbors(new_node)
                         if g[new_node][u]['sign'] > 0
                         and u in prev_nodes}
    pos_cut_to_add = {make_pair(u, new_node)
                      for u in g.neighbors(new_node)
                      if g[new_node][u]['sign'] > 0
                      and u not in prev_nodes}
    assert len(pos_cut_to_remove.intersection(pos_cut_to_add)) == 0
    new_pos_cut = prev_pos_cut - pos_cut_to_remove | pos_cut_to_add

    if verbose > 1:
        print('prev_pos_cut', prev_pos_cut)
        print('pos_cut_to_remove', pos_cut_to_remove)
        print('pos_cut_to_add', pos_cut_to_add)
        print('new_pos_cut', new_pos_cut)
        
    neg_cut_to_add = {make_pair(u, new_node)
                      for u in g.neighbors(new_node)
                      if g[new_node][u]['sign'] < 0
                      and u in prev_nodes}
    new_neg_cut = prev_neg_cut | neg_cut_to_add
    
    if verbose > 1:
        print('prev_neg_cut', prev_neg_cut)
        print('neg_cut_to_add', neg_cut_to_add)
        print('new_neg_cut', new_neg_cut)

    print('vol', new_vol)
    print('2m - vol', 2 * g.number_of_edges() - new_vol)
    denom = min(new_vol, 2 * g.number_of_edges() - new_vol)
    conductance = (2 * len(new_neg_cut) + len(new_pos_cut)) / denom
    return new_vol, new_pos_cut, new_neg_cut, conductance


def incremental_signed_conductance(g, nodes_in_order, verbose=0, show_progress=False):
    """incremental implementation of conductance computation"""
    raise Exception('deprecated, do not use')
    conductance_list = []
    prev_nodes = set()
    prev_vol = 0
    prev_pos_cut = set()
    prev_neg_cut = set()
    iter_obj = range(0, len(nodes_in_order))
    if show_progress:
        iter_obj = tqdm(iter_obj)
        
    for i in iter_obj:
        new_node = nodes_in_order[i]
        prev_vol, prev_pos_cut, prev_neg_cut, c = _one_step_for_incremental_signed_conductance(
            g, prev_nodes, new_node, prev_vol,
            prev_pos_cut, prev_neg_cut, verbose=verbose
        )
        prev_nodes |= {new_node}
        if verbose > 1:
            print('-' * 10)
        conductance_list.append(c)
    return conductance_list


def purity(g, nodes):
    """

    let g' = g.subgraph(nodes)
    purtiy(g, nodes) = #pos edges in g' / #edges in g'
    """
    subg = g.subgraph(nodes)
    signs = np.array([subg[u][v]['sign'] for u, v in subg.edges()])
    signs[signs < 0] = 0
    return signs.sum() / signs.shape[0]
    

def flatten(stuff):
    return np.asarray(stuff).flatten()


def normalized_laplacian_dense(A):
    deg = A.sum(axis=0)
    D_neg_half = np.diag(flatten(1 / np.sqrt(deg)))
    L_norm = np.eye(A.shape[0]) - D_neg_half.dot(A).dot(D_neg_half)
    return L_norm


def parse_graph_in_csv(path, sep=' ', verbose=0):
    df = pd.read_csv(
        path,
        sep=sep,
        header=None,
        names=['u', 'v', 'sign'],
        comment='%'
    )
    g = nx.Graph()

    for _, (u, v, sign) in df.iterrows():
        if g.has_edge(u, v):
            if g[u][v]['sign'] != sign:
                if verbose > 0:
                    print('conflicting sign for edge ({}, {}), remove it'.format(u, v))
                g.remove_edge(u, v)
        g.add_edge(u, v)
        g[u][v]['sign'] = sign
    g.remove_edges_from(nx.selfloop_edges(g))
    return nx.convert_node_labels_to_integers(g)


def get_lcc(g):
    """get largest connected component"""
    cc_list = nx.connected_component_subgraphs(g)
    lcc = max(cc_list, key=lambda cc: cc.number_of_nodes())
    return lcc


def laplacian(A):
    deg = A.sum(axis=0)
    L = sp.diags(flatten(deg)) - A
    return L


def normalized_laplacian(A, subtract=True):
    """
    subtract=False maps to Q case
    """
    deg = A.sum(axis=0)
    D_neg_half = sp.diags(flatten(1 / np.sqrt(deg)))
    part1 = sp.eye(A.shape[0])
    part2 = D_neg_half @ A @ D_neg_half
    if subtract:
        return part1 - part2
    else:
        return part1 + part2


def signed_normalized_laplacian(A):
    deg = abs(A).sum(axis=0)
    D_neg_half = sp.diags(flatten(1 / np.sqrt(deg)))

    L_norm = sp.eye(A.shape[0]) - D_neg_half @ A @ D_neg_half
    return L_norm
    

def conductance(g, S, weight=None, verbose=False):
    """unsigned conductance, taking into account edge weight"""
    numer = 0
    denum = 0
    S = set(S)
    vol = sum(d for _, d in g.degree(weight='weight'))
    # vol -= sum(g[u][u]['weight'] for u, u in g.selfloop_edges())
    if verbose >= 1:
        print('total vol', vol)
    for u in S:
        for v in g.neighbors(u):
            w = g[u][v].get('weight', 1)
            if v not in S:
                numer += w
            denum += w
    if verbose >= 1:
        print('{} / min({}, {})'.format(numer, denum, vol - denum))
    denum = min(denum,  vol - denum)
    return numer / denum


def dict2array(d):
    a = np.zeros(len(d), dtype=float)
    for i, v in d.items():
        a[i] = v
    return a


def signed_group_conductance(g, groups, verbose=0):
    """
    conductance generlized to a bag of node sets

    groups: a bag of node sets, denoted as S

    ( \sum_{i \neq j} \partial^{+}(S_i, S_j) + \sum_i \partial^{-}(S_i) ) / \sum_i vol(S_i)
    """
    numer_sum = 0
    denum_sum = 0
    for S in groups:
        numer = 0
        denum = 0
        S = set(S)
        for u in S:
            for v in g.neighbors(u):
                if v in S and g[u][v]['sign'] == -1:
                    numer += 1
                elif v not in S and g[u][v]['sign'] == 1:
                    numer += 1
                denum += 1
        if verbose >= 1:
            print('{} / {}'.format(numer, denum))
        numer_sum += numer
        denum_sum += denum
    if verbose >= 1:
        print('total: {} / {}'.format(numer_sum, denum_sum))

    return numer_sum / denum_sum


def get_borderless_fig():
    fig, ax = plt.subplots(1, 1)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    return fig, ax


def conductance_vectorized(C, S, self_degree, verbose=0):
    """
    get incidence matrix C using:
        C = nx.incidence_matrix(g, weight=weight).T
    get self_degree using:
        A = nx.adjacency_matrix(g, weight='weight')
        self_degree = A.diagonal()
    """
    total_vol = C.sum() + self_degree.sum()
    if verbose > 0:
        print('vol_by_self', self_degree.sum())
        print('total_vol', total_vol)
    
    C_S = C[:, S]
    weights = flatten(C_S.sum(axis=1))
    vol = weights.sum()
    vol += self_degree[S].sum()
    
    C_S = C_S[weights > 0, :]
    cut_mask = (flatten(C_S.astype('bool').sum(axis=1)) == 1)
    cut_size = C_S[cut_mask, :].sum()
    if verbose > 0:
        print('{} / min({}, {})'.format(cut_size, vol, total_vol - vol))
    return cut_size / min(vol, total_vol - vol)


def signed_conductance_by_sweeping(A, order):
    """vectorized sweeping (signed conductance)"""
    # relevant adj matrices
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    pos_A.eliminate_zeros()

    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    neg_A.eliminate_zeros()
    neg_A = -neg_A

    # negative part
    neg_B = neg_A[order, :][:, order]  # permute the matrix, both rows and columns
    neg_B_sums = flatten(neg_B.sum(axis=1))
    neg_volumes = np.cumsum(neg_B_sums)

    neg_B_lower = sp.tril(neg_B)
    neg_B_lower_sums = flatten(neg_B_lower.sum(axis=1))
    neg_penalty = np.cumsum(2 * neg_B_lower_sums)
    neg_cut = np.cumsum(neg_B_sums - 2 * neg_B_lower_sums)
    neg_penalty_other = neg_A.sum() - 2 * neg_cut - neg_penalty

    # positive part
    pos_B = pos_A[order, :][:, order]  # permute the matrix, both rows and columns
    pos_B_lower = sp.tril(pos_B)
    pos_B_sums = flatten(pos_B.sum(axis=1))
    pos_volumes = np.cumsum(pos_B_sums)  # pos_volumes
    
    # distinguish diagonal entries and non-diagonal ones
    pos_self_degree = pos_B.diagonal()
    pos_B_lower_off_diag = pos_B_lower - sp.diags(pos_self_degree)
    pos_B_lower_off_diag_sums = flatten(pos_B_lower_off_diag.sum(axis=1))
    
    pos_penalty = np.cumsum(pos_B_sums - 2 * pos_B_lower_off_diag_sums - pos_self_degree)  # to fix
    
    # together
    total_vol = abs(A).todense().sum()
    volumes = pos_volumes + neg_volumes
    volumes_other = total_vol * np.ones(len(order)) - volumes
    
    neg_penalty_selected = scipy.where(volumes < volumes_other, neg_penalty, neg_penalty_other)
    vols = np.minimum(volumes, volumes_other)
    scores = (pos_penalty + neg_penalty_selected) / vols
    return scores


def conductance_by_sweeping(A, order):
    """return n conductance scores
    where the ith entry considers the conductance of the subgraph of nodes order_0...order_i"""
    B = A[order, :][:, order]  # permute the matrix
    B_lower = sp.tril(B)
    B_sums = flatten(B.sum(axis=1))
    volumes = np.cumsum(B_sums)

    # distinguish diagonal entries and non-diagonal ones
    self_degree = B.diagonal()
    B_lower_off_diag = B_lower - sp.diags(self_degree)
    B_lower_off_diag_sums = flatten(B_lower_off_diag.sum(axis=1))

    num_cut = np.cumsum(B_sums - 2 * B_lower_off_diag_sums - self_degree)  # to fix

    total_vol = A.sum()
    volumes_other = total_vol * np.ones(len(order)) - volumes
    vols = np.minimum(volumes, volumes_other)

    # print('num_cut', num_cut)
    # print('vols', vols)

    scores = num_cut / vols
    return scores


def labels2groups(labels):
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[l].append(i)
    return groups


def n_neg_edges(g):
    return sum((g[u][v]['sign'] < 0) for u, v in g.edges())


def n_pos_edges(g):
    return sum((g[u][v]['sign'] > 0) for u, v in g.edges())


def num_ccs(g):
    return len(list(nx.connected_components(g)))


def cc_sizes(g):
    return list(sorted(map(len, nx.connected_components(g)), reverse=True))


def motif_primary_key_value(graph_path, motif_ids, teleport_alpha, query_node):
    """get hash as value of primary key"""
    msg = "{},{},{},{}".format(graph_path, ''.join(motif_ids), teleport_alpha, query_node)
    return hashlib.sha224(msg.encode('utf8')).hexdigest()


def node2connected_component(cc_list):
    n2cc = {}
    for i, cc in enumerate(cc_list):
        for n in cc:
            n2cc[n] = i
    return n2cc


def approx_diameter(g):
    """factor-2 approximation of diameter"""
    n = random.choice(list(g.nodes()))
    dist = nx.shortest_path_length(g, source=n)
    return max(list(dist.values()))
    

def num_good_edges(g):
    return np.array([g[u][v]['label'] for u, v in g.edges()]).sum()


def num_bad_edges(g):
    return g.number_of_edges() - num_good_edges(g)


def noise_level(g, weight=None):
    """fraction of 'bad' edges of all edges"""
    if not weight:
        return num_bad_edges(g) / g.number_of_edges()
    else:
        good_edge_weights = np.array([g[u][v]['label'] * abs(g[u][v][weight]) for u, v in g.edges()])
        all_edge_weights = np.array([abs(g[u][v][weight]) for u, v in g.edges()])
        return 1 - good_edge_weights.sum() / all_edge_weights.sum()


def pos_adj(A):
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    pos_A.eliminate_zeros()
    return pos_A


def neg_adj(A):
    """entry values are positive"""
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    neg_A.eliminate_zeros()
    return -neg_A


def degree_diag(g):
    """diagonal matrix with degrees as the diagonal"""
    deg = nx.adjacency_matrix(g).sum(axis=0)
    return diags(deg.tolist()[0], 0)


def prepare_seed_vector(seeds, D):
    """
    prepare seed vector s s.t.
    s.T D s = 1

    assuming one or two seeds are given,
    if two, they're in opposing communities
    """
    n = D.shape[0]
    s = np.zeros(n)
    
    for u in seeds[0]:
        s[u] = 1

    if len(seeds) > 1:
        # has 2nd seed
        for u in seeds[1]:
            s[u] = -1

    s /= np.linalg.norm(s, 2)  # l2 norm
    s = s[:, None]
    
    s = np.diag(1 / np.sqrt(D.diagonal())) @ s
    
    # requirement check
    sTDs = (s.T @ D @ s)
    assert np.isclose(sTDs[0, 0], 1.0), '{} != 1.0'.format(sTDs[0, 0])
    return s


def prepare_seed_vector_sparse(seeds, D, verbose=0):
    """
    sparse version of preparing seed vector s s.t.
    s.T D s = 1

    assuming seeds from one or two communities are given,
    if two, the communities are opposing each other
    """

    n = D.shape[0]

    i = list(seeds[0])
    j = ([0] * len(seeds[0]))
    data = ([1.0] * len(seeds[0]))

    if len(seeds) > 1:
        i += list(seeds[1])
        j += ([0] * len(seeds[1]))
        data += ([-1.0] * len(seeds[1]))

    s = coo_matrix((data, (i, j)), shape=(n, 1), dtype=np.float64)
    s /= norms(s, ord=2, axis=0)[0]  # l2 norm

    assert (D.diagonal() > 0).all()

    s = diags(1 / np.sqrt(D.diagonal())) @ s

    # requirement check
    sTDs = (s.T @ D @ s).A[0, 0]
    assert np.isclose(sTDs, 1.0), '{} != 1.0'.format(sTDs)
    if verbose > 0:
        print('s', s.A)
        print('s.shape', s.shape)
    return s


def effective_rank(A, abs_tol=1e-6):
    """compute the effective rank using SVD"""
    _, sigma, _ = svd(A)
    return (np.absolute(sigma) > abs_tol).nonzero()[0].shape[0]


def is_rank_one(M, verbose=False):
    """
    np.linalg.matrix_rank(np.round(X.value, 3)) does not work because of imperfect numeraical precision
    """
    return effective_rank(M) == 1


def sbr(A, S1, S2, verbose=0, return_details=False):
    """
    compute signed bipartiteness ratio given two sets S1 and S2

    A: adj matrix
    """
    S = list(set(S1) | set(S2))
    if verbose > 0:
        print('S1', S1)
        print('S2', S2)
    neg_degree_inside = ((A[S1, :][:, S1] < 0).sum() + (A[S2, :][:, S2] < 0).sum())
    pos_degree_between = (A[S1, :][:, S2] > 0).sum() * 2
    vol_inside = scipy.absolute(A[S, :][:, S]).sum()
    vol_total = scipy.absolute(A[S, :]).sum()
    edges_outside = vol_total - vol_inside

    if verbose > 0:
        print('neg_degree_inside=', neg_degree_inside)
        print('pos_degree_between=', pos_degree_between)
        print('vol_total=', vol_total)
        print('edges_outside=', edges_outside)

    if verbose > 1:
        print('A', A.A)
        print('S', S)
        print('A[S, :].A', A[S, :].A)
    # TODO: should be multiplied by 2?
    val = (edges_outside + neg_degree_inside + pos_degree_between) / vol_total
    math_str = "({}+{}+{})/{} = {}".format(
        edges_outside,
        neg_degree_inside,
        pos_degree_between,
        vol_total,
        val
    )

    assert val >= 0 and val <= 1, "out of range, {}".format(math_str)
    details = dict(
        edges_outside=edges_outside,
        neg_degree_inside=neg_degree_inside,
        pos_degree_between=pos_degree_between,
        vol_total=vol_total,
        math_str=math_str
    )
    if return_details:
        return val, details
    else:
        return val


def sbr_by_threshold(g, x, t):
    """
    compute signed bipartiteness ratio using the rounding result by threshold `t`
    g: graph
    x: score vector
    t: threshold
    """
    assert t >= 0, 't={}'.format(t)
    A = nx.adj_matrix(g, weight='sign')
    
    S1 = np.nonzero(x <= -t)[0]
    S2 = np.nonzero(x >= t)[0]
    return sbr(A, S1, S2, verbose=0)


def get_theoretical_kappa(S, seeds, A):
    """get the theoretic kappa defined by a set and seed nodes,
    A is the adj matrix"""
    vol_S = scipy.absolute(A[S, :]).sum()
    vol_uv = scipy.absolute(A[flatten(seeds), :]).sum()
    k = vol_S / vol_uv
    return np.sqrt(1/k)


def sample_seeds(true_comms, true_groupings, target_comm=None, k=1):
    if target_comm is None:
        target_comm = np.random.choice(len(true_comms))

    if k > len(true_groupings[target_comm][0]) or k > len(true_groupings[target_comm][1]):
        raise ValueError('k is too large')
    
    v1 = np.random.permutation(true_groupings[target_comm][0])[:k]
    v2 = np.random.permutation(true_groupings[target_comm][1])[:k]
    seeds = [v2, v1]
    return seeds, target_comm


def pos_nbrs(g, nn):
    """common neighbours count as multiple times"""
    if isinstance(nn, int):
        nn = [nn]
    nbrs = [
        i
        for n in nn
        for i in g[n]
        if g[n][i]['sign'] > 0 and i not in nn
    ]
    return nbrs


def neg_nbrs(g, nn):
    """common neighbours count as multiple times"""
    if isinstance(nn, int):
        nn = [nn]
    nbrs = [
        i
        for n in nn
        for i in g[n]
        if g[n][i]['sign'] < 0 and i not in nn
    ]
    return nbrs


def num_pos_edges(g):
    return sum(1 for u, v in g.edges() if g[u][v]['sign'] > 0)


def num_neg_edges(g):
    return sum(1 for u, v in g.edges() if g[u][v]['sign'] < 0)


def get_v1(g):
    """return the bottom-most eigen vector"""
    A = nx.adjacency_matrix(g, weight='sign')
    L = signed_normalized_laplacian(A)
    eig_val, eig_vec = eigs(L, k=1, which='SM')
    return np.real(eig_val[0]), np.real(flatten(eig_vec))


def sample_nodes_by_log_of_degree(D, size):
    n = D.shape[0]
    deg = D.diagonal()
    p = np.log2(deg.copy())
    p /= p.sum()
    sample_seeds = np.random.choice(np.arange(n), size=size, replace=False, p=p)
    sys.stderr.write('degree mean of samples: {}\n'.format(np.mean(deg[sample_seeds])))
    return sample_seeds


def extract_ijvmn(M, use_matlab=False):
    """extract rows, columns, data, m and n of a sparse matrix M """
    i, j = M.nonzero()
    v = flatten(M[i, j])

    if use_matlab:
        # matlab is 1-indexed
        i += 1
        j += 1
    m, n = M.shape
    return i, j, v, m, n


def neg_graph(g):
    A = nx.adj_matrix(g, weight='sign')
    neg_A = neg_adj(A)
    return nx.from_scipy_sparse_matrix(neg_A)


def pos_graph(g):
    A = nx.adj_matrix(g, weight='sign')
    pos_A = pos_adj(A)
    return nx.from_scipy_sparse_matrix(pos_A)
