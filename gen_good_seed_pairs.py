# coding: utf-8
"""
generate good seed pairs and dump it into file

a good seed pair (u, v) satisfy:

- there is a negative edge between u and v
- both u and v has at least {threshold} positive degree

the threshold are specified according to th  graph:

- word, bitcoin: 5
- ref: 15
- epinions, slashdot, wikiconflict: 20

the resulting number of pairs in each graph:

- word: 8604
- bitcoin: 1323
- ref: 9286
- epinions: 36577
- slashdot: 43373
- wikiconflict: 87971

"""
import sys
import networkx as nx
import pickle as pkl

from tqdm import tqdm

from helpers import pos_graph, neg_graph


graph = sys.argv[1]
threshold = int(sys.argv[2])

g = nx.read_gpickle('graphs/{}.pkl'.format(graph))


pos_g = pos_graph(g)
neg_g = neg_graph(g)

pos_deg = pos_g.degree()

good_pairs = []
for u, v in tqdm(neg_g.edges(), total=neg_g.number_of_edges()):
    if pos_deg[u] >= threshold and pos_deg[v] >= threshold:
        good_pairs.append((u, v))


print('found {} good pairs from {} candidates'.format(len(good_pairs), neg_g.number_of_edges()))

with open('outputs/{}_good_pairs.pkl'.format(graph), 'wb') as f:
    pkl.dump(good_pairs, f)

