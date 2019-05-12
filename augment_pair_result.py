"""
augment our result (seed pairs) by various statistics


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
if len(sys.argv) > 2:
    k = int(sys.argv[2])
    print('restricting k={}'.format(k))
else:
    print('dropping k')
    k = None

g = nx.read_gpickle('graphs/{}.pkl'.format(graph))
A = nx.adj_matrix(g, weight='sign')
df = pd.read_pickle('outputs/{}_seed_pair.pkl'.format(graph))

if k is not None:
    df = df[df['k'] == k]
print(df.shape[0], ' rows')
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

df['q1deg'] = df['seed1'].apply(lambda n: g.degree(n))
df['q2deg'] = df['seed2'].apply(lambda n: g.degree(n))

df = populate_fields(df, pos_A, neg_A)

if k is not None:
    df.to_pickle('outputs/{}_seed_pair_aug_k{}.pkl'.format(graph, k))
else:
    df.to_pickle('outputs/{}_seed_pair_aug.pkl'.format(graph))
