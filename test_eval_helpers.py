import pytest
import networkx as nx
from eval_helpers import (
    edge_agreement_ratio,
    evaluate_level_1,
    evaluate_level_2
)


@pytest.fixture
def g():
    graph_name = 'toy'
    path = 'graphs/{}.pkl'.format(graph_name)
    return nx.read_gpickle(path)


def test_edge_agreement_ratio(g):
    A = nx.adjacency_matrix(g, weight='sign')

    # case 1
    nodes = [0, 1, 2]
    subg = g.subgraph(nodes)
    
    expected = 2 / 8
    actual = edge_agreement_ratio(subg, A)
    assert expected == actual

    # case 2
    nodes = [0, 1, 3]
    subg = g.subgraph(nodes)

    actual = edge_agreement_ratio(subg, A)
    expected = 6 / 8
    assert expected == actual


def test_evaluate_level_1():
    n = 10
    C_pred = [0, 1, 2, 3]
    C_true = [1, 2, 3, 4, 5]
    p, r, f, _ = evaluate_level_1(n, C_pred, C_true)
    assert p == 3/4
    assert r == 3/5


def test_evaluate_level_2():
    n = 10
    groups = [[0, 1], [2, 3]]
    C_true = [0, 1, 2, 3]
    # label invariant check
    for c1, c2 in [([0, 1], [2, 3]), ([2, 3], [0, 1])]:
        p, r, f, _ = evaluate_level_2(n, c1, c2, C_true, groups)
        assert p == 1
        assert r == 1
        assert f == 1

    c1, c2 = [0, 1], [2, 3]
    groups = [[0, 3], [2, 1]]
    C_true = [0, 1, 2, 3]
    p, r, f, _ = evaluate_level_2(n, c1, c2, C_true, groups)
    assert p == 0.5
    assert r == 0.5
    assert f == 0.5
    
   
 
