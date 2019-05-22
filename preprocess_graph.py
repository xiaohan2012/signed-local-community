# coding: utf-8

import networkx as nx
import pandas as pd
from tqdm import tqdm
from helpers import get_lcc, get_v1


graph = 'bitcoin'

df = pd.read_csv('data/{}.txt'.format(graph), sep='\t', comment='#', header=None, names=['u', 'v', 'sign'])


g = nx.Graph()
dg = nx.DiGraph()
for i, r in tqdm(df.iterrows()):
    u, v, sign = r['u'], r['v'], r['sign']
    dg.add_edge(u, v, sign=sign)    
    if not g.has_edge(u, v):
        g.add_edge(u, v, sign=sign)
    else:
        print('edge ({}, {}) exists'.format(u, v))


lcc = get_lcc(g)
mapping = {n: i for i, n in enumerate(lcc.nodes())}

g = nx.relabel_nodes(lcc, mapping=mapping)
dg = nx.relabel_nodes(dg.subgraph(g.nodes()), mapping=mapping)


l1, v1 = get_v1(g)
g.graph['lambda1'] = l1
g.graph['v1'] = v1

nx.write_gpickle(g, 'graphs/{}.pkl'.format(graph))
nx.write_gpickle(dg, 'graphs/{}_directed.pkl'.format(graph))

