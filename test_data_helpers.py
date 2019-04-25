import networkx as nx
import random
import numpy as np

from data_helpers import make_polarized_graphs_fewer_parameters


def test_make_polarized_graphs_fewer_parameters__no_extra_nodes():
    """
    no extra nodes are included
    """
    random.seed(12345)
    np.random.seed(2345)

    # case 1
    # no noise, eta=0
    nc, eta = 100, 0
    g, _, true_groupings = make_polarized_graphs_fewer_parameters(
        nc, 0, 1, eta
    )
    A = nx.adjacency_matrix(g, weight='sign')
    C1, C2 = true_groupings[0]
    for C in [C1, C2]:
        assert A.A[C, :][:, C].sum() == nc * (nc-1)
    assert (-A.A[C1, :][:, C2]).sum() == nc * nc
    assert (-A.A[C2, :][:, C1]).sum() == nc * nc

    # case 2
    # all noise, eta=1
    nc, eta = 100, 1.0
    g, _, true_groupings = make_polarized_graphs_fewer_parameters(nc, 0, 1, eta)
    A = nx.adjacency_matrix(g, weight='sign')
    C1, C2 = true_groupings[0]
    for C in [C1, C2]:
        neg_ratio = -A.A[C, :][:, C].sum() / nc / (nc-1)  # half are neg edges
        assert np.isclose(neg_ratio, 0.5, rtol=0.05), neg_ratio
    # in between, all positive edges
    A_sub = A.A[C1, :][:, C2]
    assert (A_sub >= 0).all()
    # around half are created
    # the other half are non-existing
    ratio = A_sub.sum() / (nc * nc / 2)
    assert np.isclose(ratio, 1, rtol=0.01), ratio

    # case 3
    # half noise, eta=0.5
    nc, eta = 100, 0.5
    g, _, true_groupings = make_polarized_graphs_fewer_parameters(nc, 0, 1, eta)
    A = nx.adjacency_matrix(g, weight='sign')
    C1, C2 = true_groupings[0]
    for C in [C1, C2]:
        A_sub = A.A[C, :][:, C]
        pos_degree = A_sub[A_sub > 0].sum()
        neg_degree = abs(A_sub[A_sub < 0].sum())
        pos_ratio = pos_degree / nc / (nc-1)
        neg_ratio = neg_degree / nc / (nc-1)
        assert np.isclose(pos_ratio, 2/4, rtol=0.1), pos_ratio
        assert np.isclose(neg_ratio, 1/4, rtol=0.1), neg_ratio
        
    A_sub = A.A[C1, :][:, C2]
    pos_degree = A_sub[A_sub > 0].sum()
    neg_degree = abs(A_sub[A_sub < 0].sum())
    pos_ratio = pos_degree / (nc * nc)
    neg_ratio = neg_degree / (nc * nc)

    assert np.isclose(pos_ratio, 1/4, rtol=0.05), pos_ratio
    assert np.isclose(neg_ratio, 2/4, rtol=0.05), neg_ratio


def test_make_polarized_graphs_fewer_parameters():
    """
    with extra noisy nodes
    """
    random.seed(12345)
    np.random.seed(2345)

    nc, eta, n = 100, 0.5, 100
    num_edges_before = make_polarized_graphs_fewer_parameters(
        nc, 0, 1, eta
    )[0].number_of_edges()

    g, _, true_groupings = make_polarized_graphs_fewer_parameters(
        nc, n, 1, eta
    )
    assert g.number_of_nodes() == (nc*2 + n)
    # number of possible edges times eta
    actual_num_new_egges = (g.number_of_edges() - num_edges_before)
    expected_num_new_edges = ((n * (n-1) / 2) + (n * 2 * nc)) * eta
    assert np.isclose(actual_num_new_egges/expected_num_new_edges, 1, rtol=0.01)
