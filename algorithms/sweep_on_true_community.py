import numpy as np
from .sweep_cut import get_community


def run(g, query, ground_truth):
    true_comm = next(c for c in ground_truth if query in c)
    scores = np.zeros(g.number_of_nodes())
    scores[true_comm] = 1.0 / len(true_comm)
    return get_community(g, scores)
