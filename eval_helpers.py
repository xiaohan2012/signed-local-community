import numpy as np
import networkx as nx

from itertools import chain, combinations


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
