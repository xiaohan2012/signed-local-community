import pytest
import networkx as nx
from helpers import purity, sbr


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

    
    

    
