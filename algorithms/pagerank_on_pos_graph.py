import networkx as nx
import numpy as np
from .pagerank import pr_score
from .sweep_cut import get_community


def extract_pos_graph(g):
    pos_g = nx.Graph()
    pos_g.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if g[u][v]['sign'] == 1.0:
            pos_g.add_edge(u, v)
    return pos_g


def run(g, query, alpha, return_sorted_nodes=False, verbose=0, show_progress=False):
    if verbose > 0:
        print('extracting positive graph')
    pos_g = extract_pos_graph(g)
    if verbose > 0:
        print('running pagerank')
    scores = pr_score(pos_g, query, alpha)

    if verbose > 0:
        print('running sweep cut')
    comm = get_community(g, scores, show_progress=show_progress)

    if not return_sorted_nodes:
        return comm
    else:
        return comm, np.argsort(scores)[::-1]
