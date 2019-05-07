import pytest
import random
import numpy as np
import networkx as nx
from graph_generator.community_graph import make

random.seed(12345)
np.random.seed(12345)


@pytest.fixture
def polarized_graph():
    g, groundtruth = make(4, 2, 1, 0, 0.3, 1)
    g.add_edge(8, 0, sign=1)
    g.add_edge(9, 0, sign=-1)
    g.add_edge(10, 4, sign=-1)
    g.add_edge(11, 5, sign=1)
    return g


@pytest.fixture
def toy_graph():
    """http://193.166.24.212/local-polarization-figs/test_graphs/toy.png"""
    g = nx.Graph()
    nodes = range(5)
    g.add_nodes_from(nodes)
    edges = [(0, 1, 1), (1, 2, 1), (0, 2, -1), (0, 3, 1), (2, 3, -1), (3, 4, 1)]
    for u, v, s in edges:
            g.add_edge(u, v, sign=s)
        
    return g
