import pytest
import numpy as np
import networkx as nx

from scipy.sparse import diags, issparse
from helpers import purity, sbr, prepare_seed_vector_sparse, sample_seeds


@pytest.fixture
def test_graph():
    """http://193.166.24.212/local-polarization-figs/test_graphs/toy.png"""
    g = nx.Graph()
    nodes = range(5)
    g.add_nodes_from(nodes)
    edges = [(0, 1, 1), (1, 2, 1), (0, 2, -1), (0, 3, 1), (2, 3, -1), (3, 4, 1)]
    for u, v, s in edges:
            g.add_edge(u, v, sign=s)
        
    return g


def test_purity(test_graph):
    actual = purity(test_graph, list(range(5)))
    assert actual == 2 / 3

    
def test_sbr(test_graph):
    A = nx.adj_matrix(test_graph, weight='sign')

    S1, S2 = [1], [2]
    expected = (2 + 3) / 5
    assert sbr(A, S1, S2, verbose=1) == expected

    print('-' * 10)
    S1, S2 = [0], [2]
    expected = (0 + 4) / 6
    assert sbr(A, S1, S2, verbose=1) == expected

    print('-' * 10)
    S1, S2 = [0, 2], [3, 4]
    expected = (0 + 6) / 10
    assert sbr(A, S1, S2, verbose=1) == expected


@pytest.mark.parametrize("seeds", [
    [[0, 1], [2, 3]],
    [[0, 1]],
    [[0]]
])
def test_prepare_seed_vector_sparse(seeds):
    D = diags([1, 2, 3, 4])
    
    s = prepare_seed_vector_sparse(seeds, D)
    # is sparse
    assert issparse(s)
    assert s.shape == (4, 1)
    # are active
    for seed_list in seeds:
        for seed in seed_list:
            assert not np.isclose(s[seed, 0], 0.0)
    # normalized
    assert np.isclose((s.T @ D @ s)[0, 0], 1)


def test_sample_seeds():
    true_comms = [[0, 1, 2, 3], [4, 5, 6, 7]]
    true_groupings = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]

    seeds, _ = sample_seeds(true_comms, true_groupings, k=1)
    assert len(seeds) == 2
    for s in seeds:
        assert len(set(s)) == 1
    
    seeds, _ = sample_seeds(true_comms, true_groupings, k=2)
    assert len(seeds) == 2
    for s in seeds:
        assert len(set(s)) == 2

    with pytest.raises(ValueError):
        seeds, _ = sample_seeds(true_comms, true_groupings, k=3)
