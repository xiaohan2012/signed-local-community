# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from helpers import get_lcc

g = nx.read_gpickle((sys.argv[1]))

data = []

data.append(('nodes', g.number_of_nodes()))
n_edges = g.number_of_edges()
data.append(('edges', n_edges))
num_pos_edges = sum((g[u][v]['sign'] == 1) for u, v in g.edges())
num_neg_edges = n_edges - num_pos_edges
data.append(('frac. + edges', num_pos_edges / n_edges))
data.append(('frac. - edges', num_neg_edges / n_edges))
data.append(('num. CCs', len(list(nx.connected_components(g)))))
lcc = get_lcc(g)
data.append(('frac. LCC', lcc.number_of_nodes() / g.number_of_nodes()))


# data.append(('is_directed?', g.is_directed()))

# deg = g.degree_property_map('total')

# num = int(np.sum(deg.a == 0))
# data.append(('isolated nodes', '{} ({:.2f}%)'.format(num, num / g.num_vertices() * 100)))

# labels, hist = label_components(g, directed=False)
# data.append(('number of connected components', len(hist)))

# _, hist = label_components(g, directed=False)
# hist.sort()
# if len(hist) > 1:
#     size1, size2 = hist[-1], hist[-2]
# else:
#     size1, size2 = hist[-1], 0
# data.append(('size of 1st/2nd component',
#              '{} ({:.2f}%), {}/({:.2f}%)'.format(
#                  size1, 100 * size1 / g.num_vertices(),
#                  size2, 100 * size2 / g.num_vertices())))

# data.append(('min/max/avg degree',
#              '{}/{}/{:.2f}'.format(int(deg.a.min()),
#                                    int(deg.a.max()),
#                                    float(deg.a.mean()))))

# data.append(('density', '{:.7f}'.format(2 * g.num_edges() / g.num_vertices() / (g.num_vertices() - 1))))


# data.append(('clustering coefficient (std)', '{:.2f} ({:.2f})'.format(*global_clustering(g))))


# sampled_sources = np.random.permutation(g.num_vertices())[:100]
# dist = np.max([pseudo_diameter(g, s)[0] for s in sampled_sources])
# data.append(('pseudo diameter', dist))


# data.append(('assortativity (std)', '{:.2f} ({:.2f})'.format(
#     *assortativity(g, 'total'))))

index, col = zip(*data)
s = pd.Series(col, index=index)
print(s.to_string())

# save it somewhere

# name = ''.join(os.path.basename(sys.argv[1]).split('.')[:-1])
# output_path = os.path.dirname(sys.argv[1]) + '/summary.csv'
# s.to_csv(output_path)
