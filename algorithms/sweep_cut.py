import os
import sys
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from helpers import signed_conductance, degree_array


def get_community(g, pr_scores):
    degree_normalized_pr_scores = pr_scores / degree_array(g)
    
    order = np.argsort(degree_normalized_pr_scores)[::-1]

    sweep_positions = []
    sweep_scores = []
    for i in range(1, len(order)+1):
        if degree_normalized_pr_scores[order[i-1]] == 0:
            break
        sweep_positions.append(i)
        s = signed_conductance(g, order[:i])
        sweep_scores.append(s)

    # get the best community
    best_pos = np.argmin(sweep_scores)
    comm = order[:best_pos+1]

    return list(comm)
