import pytest
import networkx as nx
from helpers import signed_conductance, incremental_conductance, purity


@pytest.fixture
def test_graph():
    g = nx.Graph()
    nodes = ['a', 'b', 'c', 'd', 'e']
    g.add_nodes_from(nodes)
    edges = [('a', 'b', 1), ('b', 'c', 1), ('a', 'c', -1), ('a', 'd', 1), ('c', 'd', -1), ('d', 'e', 1)]
    for u, v, s in edges:
        g.add_edge(u, v, sign=s)
    return g


def test_conductance_and_incremental_version(test_graph):
    nodes = ['a', 'b', 'c',  'd']
    actual = incremental_conductance(test_graph, nodes, show_progress=True)

    expected = []
    for i in range(len(nodes)):
        c = signed_conductance(test_graph, nodes[:i+1])
        expected.append(c)

    assert actual == expected

def test_purity(test_graph):
    actual = purity(test_graph, ['a', 'b', 'c', 'd', 'e'])
    assert actual == 2 / 3
