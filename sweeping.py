import networkx as nx
import numpy as np
from helpers import flatten, conductance_by_sweeping
from algorithms.pagerank import pr_score


def sweeping_scores_using_ppr(g, query, alpha, weight='weight', A=None):
    """
    run ppr and returns sweeping positions as well as scores
    """
    z_vect = pr_score(g, query, alpha)

    if A is None:
        A = nx.adjacency_matrix(g, weight=weight)

    deg = flatten(A.sum(axis=1))
    node_scores = z_vect / deg
    node_scores[np.isnan(node_scores)] = 0  # nan due to singleton nodes

    order = np.argsort(node_scores)[::-1]
    sweep_scores = conductance_by_sweeping(A, order)

    # only consider non-nan scores
    sweep_scores = sweep_scores[np.logical_not(np.isnan(sweep_scores))]
    sweep_positions = np.arange(1, len(sweep_scores)+1)
    
    return order, sweep_positions, sweep_scores
