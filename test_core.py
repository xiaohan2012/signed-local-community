import pytest
import random
import networkx as nx
import numpy as np

from itertools import combinations
from core import query_graph
from data_helpers import make_polarized_graphs
from helpers import sample_seeds, degree_diag

random.seed(12345)
np.random.seed(12345)


@pytest.mark.parametrize('rep_i', range(1))
@pytest.mark.parametrize('solver_pair', combinations(('sdp', 'sp'), 2))
def test_solver_consistency(rep_i, solver_pair):
    """
    make sure the solutions by different solvers are same (up to certain float point precsion)
    """
    solver1, solver2 = solver_pair
    size = 10
    k = 2
    g, true_comms, true_groupings = make_polarized_graphs(k, [(size, size) for i in range(k)])
    D = degree_diag(g)

    seeds, _ = sample_seeds(true_comms, true_groupings)
    
    x1, _ = query_graph(g, seeds, solver=solver1)
    x1 = x1 / np.sqrt(x1 @ D @ x1[:, None])
 
    x2, _ = query_graph(g, seeds, solver=solver2)
    x2 = x2 / np.sqrt(x2 @ D @ x2[:, None])

    ratio = np.abs(x1 / x2)
    np.isclose(np.mean(ratio), 1.0, atol=0.1)
