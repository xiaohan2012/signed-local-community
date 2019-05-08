# coding: utf-8

import networkx as nx
import scipy.io as sio
from helpers import pos_adj, neg_adj

graph_name = 'word'

g = nx.read_gpickle('graphs/{}.pkl'.format(graph_name))


adj = nx.adj_matrix(g, weight='sign')
A, B = pos_adj(adj), neg_adj(adj)

assert (A.data > 0).all()
assert (B.data > 0).all()

sio.savemat(
    'data/{}.mat'.format(graph_name),
    {'A': A, 'B': B},
    format='4'  # your might need to change this according to your Matlab version
)
