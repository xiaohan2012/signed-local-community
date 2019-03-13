import numpy as np
import networkx as nx

from itertools import chain, combinations
from helpers import n_neg_edges, n_pos_edges, approx_diameter
from collections import OrderedDict


def frac_intra_neg_edges(subg, A):
    nodes = list(subg.nodes())
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    n_neg_edges_total = np.absolute(neg_A[nodes, :].sum()) / 2
    n_neg_edges_inside = n_neg_edges(subg)
    return n_neg_edges_inside / n_neg_edges_total


def frac_inter_pos_edges(subg, A):
    nodes = list(subg.nodes())
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    n_pos_edges_total = pos_A[nodes, :].sum() / 2
    n_pos_edges_inside = n_pos_edges(subg)
    return 1 - n_pos_edges_inside / n_pos_edges_total


def edge_agreement_ratio(subg, A):
    """number of good edges / total number of edges"""
    def n_cut_and_n_inside(A, nodes):
        n_inside = A[nodes, :][:, nodes].sum() / 2
        A[:, nodes] = 0
        n_cut = A.sum()
        return n_inside, n_cut
        
    nodes = list(subg.nodes())
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    pos_A = pos_A.astype('bool')
    pos_A.eliminate_zeros()
    n_pos_edges_inside, n_pos_edges_cut = n_cut_and_n_inside(pos_A, nodes)

    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    neg_A = neg_A.astype('bool')
    neg_A.eliminate_zeros()
    n_neg_edges_inside, n_neg_edges_cut = n_cut_and_n_inside(neg_A, nodes)

    # print('n_pos_edges_inside, n_pos_edges_cut', n_pos_edges_inside, n_pos_edges_cut)
    # print('n_neg_edges_inside, n_neg_edges_cut', n_neg_edges_inside, n_neg_edges_cut)
    total = n_pos_edges_inside + n_pos_edges_cut + n_neg_edges_inside + n_neg_edges_cut
    n_good_edges = n_pos_edges_inside + n_neg_edges_cut

    agreement_ratio = n_good_edges / total
    return agreement_ratio


def community_summary(subg, g):
    A = nx.adj_matrix(g, weight='sign')
    # neg_frac = 1 - frac_intra_neg_edges(subg, A)
    # pos_frac = 1 - frac_inter_pos_edges(subg, A)
    res = OrderedDict()
    res['n'] = subg.number_of_nodes()
    res['m'] = subg.number_of_edges()
    # res['inter_neg_ratio'] = neg_frac
    # res['intra_pos_ratio'] = pos_frac
    res['edge_agreement_ratio'] = edge_agreement_ratio(subg, A)

    # res['avg_cc'] = np.mean(list(nx.clustering(subg).values()))

    # if g.number_of_nodes() <= 1000:
    #     # for small graphs, compute diameter exactly
    #     res['diameter'] = nx.diameter(subg)
    # else:
    #     res['diameter'] = approx_diameter(g)
    return res
