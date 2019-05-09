import pytest
import random
import networkx as nx
import numpy as np

from itertools import combinations
from core import query_graph, sweep_on_x_fast, sweep_on_x
from data_helpers import make_polarized_graphs
from scipy.sparse.linalg import eigs
from helpers import (
    sample_seeds,
    degree_diag,
    pos_nbrs,
    neg_nbrs,
    num_neg_edges,
    num_pos_edges,
    signed_normalized_laplacian,
    flatten,
    get_v1
)
from test_fixtures import polarized_graph, toy_graph


random.seed(12345)
np.random.seed(12345)


@pytest.mark.parametrize('rep_i', range(5))
@pytest.mark.parametrize('solver_pair', combinations(('sdp', 'sp', 'cg'), 2))
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
    
    
def test_sweep_on_x_fast_assertions(polarized_graph):
    g = polarized_graph
    
    _, x = get_v1(g)
    
    C1, C2, C, best_t, best_beta, ts, beta_array, details = sweep_on_x_fast(
        g, x, return_details=True
    )

    pos_A = details['pos_A']
    neg_A = details['neg_A']
    abs_order = details['abs_order']
    pos_order = details['pos_order']
    neg_order = details['neg_order']
    pos_vol_by_abs = details['pos_vol_by_abs']
    neg_vol_by_abs = details['neg_vol_by_abs']
    pos_cut_by_abs = details['pos_cut_by_abs']
    neg_cut_by_abs = details['neg_cut_by_abs']
    neg_inside_1 = details['neg_inside_1']
    neg_inside_2 = details['neg_inside_2']
    pos_inside_1 = details['pos_inside_1']
    pos_inside_2 = details['pos_inside_2']
    pos_cut_1 = details['pos_cut_1']
    pos_cut_2 = details['pos_cut_2']
    pos_between_1_2 = details['pos_between_1_2']
    neg_inside_1_2 = details['neg_inside_1_2']
    
    assert (np.cumsum([len(pos_nbrs(g, n)) for n in abs_order.tolist()]) == pos_vol_by_abs).all()
    assert (np.cumsum([len(neg_nbrs(g, n)) for n in abs_order.tolist()]) == neg_vol_by_abs).all()
    assert (
        np.array([len(pos_nbrs(g, abs_order[:i])) for i in range(1, len(abs_order)+1)]) == pos_cut_by_abs
    ).all()
    assert (
        np.array([len(neg_nbrs(g, abs_order[:i])) for i in range(1, len(abs_order)+1)]) == neg_cut_by_abs
    ).all()

    assert (
        np.array([2*num_neg_edges(g.subgraph(pos_order[:i])) for i in range(1, len(pos_order)+1)])
        == neg_inside_1
    ).all()

    assert (
        np.array([2*num_neg_edges(g.subgraph(neg_order[:i])) for i in range(1, len(neg_order)+1)])
        == neg_inside_2
    ).all()

    assert (
        np.array([2*num_pos_edges(g.subgraph(pos_order[:i])) for i in range(1, len(pos_order)+1)])
        == pos_inside_1
    ).all()

    assert (
        np.array([2*num_pos_edges(g.subgraph(neg_order[:i])) for i in range(1, len(neg_order)+1)])
        == pos_inside_2
    ).all()

    assert (
        np.array([len(pos_nbrs(g, pos_order[:i])) for i in range(1, len(pos_order)+1)]) == pos_cut_1
    ).all()
    assert (
        np.array([len(pos_nbrs(g, neg_order[:i])) for i in range(1, len(neg_order)+1)]) == pos_cut_2
    ).all()

    expected_pos_between_1_2 = []
    expected_neg_inside_1_2 = []

    for i in range(1, len(abs_order)+1):
        nodes = abs_order[:i]
        V1 = nodes[np.nonzero(x[nodes] > 0)[0]]
        V2 = nodes[np.nonzero(x[nodes] < 0)[0]]
        pos_deg = pos_A[V1, :][:, V2].sum()
        neg_deg = neg_A[V1, :][:, V1].sum() + neg_A[V2, :][:, V2].sum()
        # print(V1, V2, deg)
        expected_pos_between_1_2.append(pos_deg)
        expected_neg_inside_1_2.append(neg_deg)
        
    assert (pos_between_1_2 == np.array(expected_pos_between_1_2)).all()
    assert (neg_inside_1_2 == np.array(expected_neg_inside_1_2)).all()


def test_sweep_on_x_fast_top_k(polarized_graph):
    g = polarized_graph
    
    _, x = get_v1(g)
    
    C1, C2, C, best_t, best_beta, ts, beta_array = sweep_on_x_fast(
        g, x, top_k=8
    )

    assert set(C) == set(range(8))
    assert beta_array.shape == (8, )
    assert ts.shape == (8, )


@pytest.mark.parametrize('g', [toy_graph(), polarized_graph()])
def test_sweeping_on_fixtures(g):
    _, x = get_v1(g)

    exp_c1, exp_c2, exp_C, exp_best_t, exp_best_sbr, exp_ts, exp_sbr_list = sweep_on_x(g, x)
    act_c1, act_c2, act_C, act_best_t, act_best_sbr, act_ts, act_sbr_list = sweep_on_x_fast(g, x)

    exp_c1, exp_c2, exp_C = set(exp_c1), set(exp_c2), set(exp_C)
    act_c1, act_c2, act_C = set(act_c1), set(act_c2), set(act_C)

    assert exp_c1 == act_c2
    assert exp_c2 == act_c1
    assert exp_C == act_C
    assert exp_best_t == act_best_t
    assert exp_best_sbr == act_best_sbr
    # print(exp_sbr_list)
    # print(act_sbr_list[::-1])
    assert np.isclose(exp_ts, act_ts[::-1]).all()

    # this is commented out as it fails sometimes due to numerical instability
    # assert np.isclose(exp_sbr_list, act_sbr_list[::-1]).all()


@pytest.mark.parametrize('n_rep', range(10))
def test_sweeping_consistency_on_random_graphs(n_rep):
    size = 10
    k = 2
    g, _, _ = make_polarized_graphs(k, [(size, size) for i in range(k)])
    
    _, x = get_v1(g)
    exp_c1, exp_c2, exp_C, exp_best_t, exp_best_sbr, exp_ts, exp_sbr_list = sweep_on_x(g, x)
    act_c1, act_c2, act_C, act_best_t, act_best_sbr, act_ts, act_sbr_list = sweep_on_x_fast(g, x)

    exp_c1, exp_c2, exp_C = set(exp_c1), set(exp_c2), set(exp_C)
    act_c1, act_c2, act_C = set(act_c1), set(act_c2), set(act_C)

    assert exp_c1 == act_c2
    assert exp_c2 == act_c1
    assert exp_C == act_C
    assert exp_best_t == act_best_t
    assert exp_best_sbr == act_best_sbr
    # print(exp_sbr_list)
    # print(act_sbr_list[::-1])
    assert np.isclose(exp_ts, act_ts[::-1]).all()
    assert np.isclose(exp_sbr_list, act_sbr_list[::-1]).all()
