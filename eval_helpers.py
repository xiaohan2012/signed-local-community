import numpy as np
import networkx as nx

from itertools import chain, combinations
from helpers import n_neg_edges, n_pos_edges, approx_diameter
from collections import OrderedDict


def edge_agreement_ratio(g, groups):
    nodes = list(chain(*groups))
    g_sub = g.subgraph(nodes)
    n_edges = g_sub.number_of_edges()
    node_label = {u: i for i, grp in enumerate(groups) for u in grp}
    num_agrees = 0
    for u, v in g_sub.edges():
        if g_sub[u][v]['sign'] == 1:
            num_agrees += int(node_label[u] == node_label[v])
        else:
            num_agrees += int(node_label[u] != node_label[v])
    return num_agrees / n_edges


def avg_cc(g, groups):
    cc_list = []
    for grp in groups:
        subg = g.subgraph(grp).copy()
        subg.remove_edges_from([(u, v) for u, v in subg.edges() if subg[u][v]['sign'] < 0])
        cc_list += list(nx.clustering(subg).values())
    return np.mean(cc_list)


def cohesion(g, grp):
    subg = g.subgraph(grp).copy()
    subg.remove_edges_from([(u, v) for u, v in subg.edges() if subg[u][v]['sign'] < 0])
    n = subg.number_of_nodes()
    return 2 * subg.number_of_edges() / n / (n-1)


def avg_cohesion(g, groups):
    return np.mean([cohesion(g, grp) for grp in groups])


def opposition(g, grp1, grp2):
    grp1, grp2 = map(set, [grp1, grp2])
    subg = g.subgraph(grp1 | grp2)

    cnt = 0
    for u, v in subg.edges():
        if (u in grp1 and v in grp2) or (u in grp2 and v in grp1):
            cnt += int(subg[u][v]['sign'] < 0)
    total = len(grp1) * len(grp2)
    return cnt / total


def avg_opposition(g, groups):
    return np.mean(
        [opposition(g, grp1, grp2) for grp1, grp2 in combinations(groups, 2)]
    )


def opposing_groups_summary(g, groups):
    return dict(
        agree_ratio=edge_agreement_ratio(g, groups),
        cc=avg_cc(g, groups),
        coh=avg_cohesion(g, groups),
        opp=avg_opposition(g, groups)
    )


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


def community_summary(subg, g):
    A = nx.adj_matrix(g, weight='sign')
    neg_frac = 1 - frac_intra_neg_edges(subg, A)
    pos_frac = 1 - frac_inter_pos_edges(subg, A)
    res = OrderedDict()
    res['n'] = subg.number_of_nodes()
    res['m'] = subg.number_of_edges()
    res['inter_neg_edges'] = neg_frac
    res['intra_pos_edges'] = pos_frac
    res['f1_pos_neg'] = 2 * (pos_frac * neg_frac) / (pos_frac + neg_frac)

    res['avg_cc'] = np.mean(list(nx.clustering(subg).values()))

    if g.number_of_nodes() <= 1000:
        # for small graphs, compute diameter exactly
        res['diameter'] = nx.diameter(subg)
    else:
        res['diameter'] = approx_diameter(g)
    return res
