import numpy as np
import random
import string
import datetime

import pandas as pd
import networkx as nx

from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy import sparse as sp
from matplotlib import pyplot as plt
from tqdm import tqdm


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


def signed_layout(g):
    L = signed_laplacian(g)
    w, pos_array = eigs(L.asfptype(), k=2, which='SM')
    pos_array = np.real(pos_array)
    return {i: pos_array[i, :] for i in range(g.number_of_nodes())}


def draw_nodes(g, pos, labels=None, ax=None):
    nx.draw_networkx_nodes(g, pos, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax)


def draw_edges(g, pos, ax=None):
    pos_edges = [(u, v) for u, v in g.edges() if g[u][v]['sign'] == 1.0]
    neg_edges = [(u, v) for u, v in g.edges() if g[u][v]['sign'] == -1.0]
    nx.draw_networkx_edges(g, pos, pos_edges, style='solid', edge_color='blue', ax=ax)
    nx.draw_networkx_edges(g, pos, neg_edges, style='dashed', edge_color='red', ax=ax)


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


def _one_step_for_incremental_conductance(
        g, prev_nodes, new_node, prev_vol, prev_pos_cut, prev_neg_cut, verbose=2
):
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


def incremental_conductance(g, nodes_in_order, verbose=0, show_progress=False):
    """incremental implementation of conductance computation"""
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
        prev_vol, prev_pos_cut, prev_neg_cut, c = _one_step_for_incremental_conductance(
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


def normalized_laplacian(A):
    deg = A.sum(axis=0)
    D_neg_half = sp.diags(flatten(1 / np.sqrt(deg)))
    L_norm = sp.eye(A.shape[0]) - D_neg_half @ A @ D_neg_half
    return L_norm


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
    for u in S:
        for v in g.neighbors(u):
            w = g[u][v].get('weight', 1)
            if v not in S:
                numer += w
            denum += w
    denum = min(denum,  vol - denum)
    if verbose >= 1:
        print('{} / {}'.format(numer, denum))
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
