"""
transform the graph by adding:
- a hub
- interception node for each negative edge

and then run pagerank on the transformed graph
"""
import numpy as np
import networkx as nx
from .pagerank import pr_score
from .sweep_cut import get_community


def transform_graph(g, beta, gamma, relabel=True):
    assert beta >= 1
    assert gamma <= 1 and gamma >= 0
    gp = g.to_directed()
    hub_node = 'h'
    gp.add_node(hub_node)

    original_nodes = list(gp.nodes())
    original_edges = list(gp.edges())

    for u, v in original_edges:
        if g[u][v]['sign'] == 1:
            gp[u][v]['weight'] = beta
        else:
            assert g[u][v]['sign'] == -1
            i = '{}->{}'.format(u, v)
            gp.remove_edge(u, v)
            gp.add_node(i)
            gp.add_edge(u, i, weight=1)
            if gamma > 0:
                gp.add_edge(i, v, weight=gamma)
            gp.add_edge(i, hub_node, weight=1)

    for u in original_nodes:
        gp.add_edge(hub_node, u, weight=1)
    if relabel:
        return nx.convert_node_labels_to_integers(gp)
    else:
        return gp


def run(g, query, alpha, beta=10, gamma=0, return_sorted_nodes=False):
    N = g.number_of_nodes()
    gp = transform_graph(g, beta, gamma)
    scores = pr_score(gp, query, alpha)[:N]
    if not return_sorted_nodes:
        return get_community(g, scores)
    else:
        return get_community(g, scores), np.argsort(scores)[::-1]
