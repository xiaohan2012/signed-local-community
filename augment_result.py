"""
augment our result by various statistics


exclude those that:

- C1 or C2 is small (<{threshold} nodes)

"""
import sys
import pandas as pd
import numpy as np
import networkx as nx

from helpers import (
    pos_adj, neg_adj
)
from stat_helpers import populate_fields


graph = sys.argv[1]

g = nx.read_gpickle('graphs/{}.pkl'.format(graph))
A = nx.adj_matrix(g, weight='sign')
df = pd.read_pickle('outputs/{}.pkl'.format(graph))

pos_A, neg_A = pos_adj(A), neg_adj(A)

# for weighted matrix, make it unweighted
if not (pos_A.data == 1).all():
    pos_A.data = np.ones(pos_A.data.shape, dtype='float64')
if not (neg_A.data == 1).all():
    neg_A.data = np.ones(neg_A.data.shape, dtype='float64')

threshold = 5
# k_value = 200
# df = df[df['k'] == k_value]

df['size1'] = df['C1'].apply(lambda d: d.shape[0])
df['size2'] = df['C2'].apply(lambda d: d.shape[0])
df['balancedness'] = np.minimum(df['size1'], df['size2']) / (df['size1'] + df['size2'])

df = df[(df['size1'] > threshold) & (df['size2'] > threshold)]

df['qdeg'] = df['query'].apply(lambda n: g.degree(n))
df = populate_fields(df, pos_A, neg_A)

df.to_pickle('outputs/{}_aug.pkl'.format(graph))
