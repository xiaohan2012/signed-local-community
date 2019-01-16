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


def run(
        g, query, alpha,
        beta=10, gamma=0,
        truncate_percentile=0,
        return_sorted_nodes=False,
        verbose=0, show_progress=False
):
    """
    - `alpha`: probability to teleport
    - `beta` ([0, \infty]): amplification factor to positive edges against negative ones
    - `gamma` ([0, 1]): deamplification factor to negative edges against non edges
    - `truncate_percentile` specifies the maximum number of nodes to consider in the sweep sets

    """
    N = g.number_of_nodes()
    if verbose > 0:
        print('transforming graph')
    gp = transform_graph(g, beta, gamma)

    if verbose > 0:
        print('running pagerank')
    scores = pr_score(gp, query, alpha)[:N]

    assert truncate_percentile >= 0 and truncate_percentile <= 100
    score_threshold = np.percentile(scores, truncate_percentile)
    if verbose > 0:
        print('score_threshold {} (percentile={})'.format(
            score_threshold, truncate_percentile)
        )
    
    if verbose > 0:
        print('running sweep cut')
    comm = get_community(g, scores, pr_threshold=score_threshold, show_progress=show_progress)

    if not return_sorted_nodes:
        return comm
    else:
        return comm, np.argsort(scores)[::-1]
