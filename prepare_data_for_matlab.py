# coding: utf-8

import networkx as nx
import scipy.io as sio
from helpers import pos_adj, neg_adj, extract_ijvmn

graph_name = 'wikiconflict'

g = nx.read_gpickle('graphs/{}.pkl'.format(graph_name))


adj = nx.adj_matrix(g, weight='sign')
A, B = pos_adj(adj), neg_adj(adj)

assert (A.data > 0).all()
assert (B.data > 0).all()

ai, aj, av, am, an = extract_ijvmn(A, use_matlab=True)
bi, bj, bv, bm, bn = extract_ijvmn(B, use_matlab=True)

# for "word", change format to  "5"
sio.savemat(
    '../KOCG.SIGKDD2016/DATA/{}.mat'.format(graph_name),
    dict(
        ai=ai, aj=aj, av=av, am=am, an=an,
        bi=bi, bj=bj, bv=bv, bm=bm, bn=bn
    )
    # format='4'  # your might need to change this according to your Matlab version
)
