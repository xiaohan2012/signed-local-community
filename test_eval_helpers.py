import pytest
import networkx as nx
from eval_helpers import edge_agreement_ratio


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
