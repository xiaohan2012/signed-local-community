import pytest
import numpy as np
import networkx as nx

from scipy.sparse import diags, issparse
from helpers import purity, sbr, prepare_seed_vector_sparse, sample_seeds
from test_fixtures import toy_graph as test_graph


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
