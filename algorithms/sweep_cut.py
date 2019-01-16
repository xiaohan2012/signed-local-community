import os
import sys
import numpy as np
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from helpers import degree_array, incremental_conductance


def get_community(g, pr_scores, pr_threshold=0.0, verbose=0, show_progress=False):
    if pr_threshold > 0:
        pr_scores[pr_scores <= pr_threshold] = 0.0  # truncate nodes with small PR values
    degree_normalized_pr_scores = pr_scores / degree_array(g)

    nnz = len(np.nonzero(degree_normalized_pr_scores)[0])
    # take only nnz nodes
    nodes_in_order = np.argsort(degree_normalized_pr_scores)[::-1][:nnz]

    sweep_scores = incremental_conductance(
        g, nodes_in_order, verbose=verbose,
        show_progress=show_progress
    )
    
    # get the best community
    best_pos = np.argmin(sweep_scores)
    comm = nodes_in_order[:best_pos+1]

    return list(comm)
