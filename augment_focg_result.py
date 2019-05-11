"""
pre-process the output by FOCG (KDD, 2018) and augment the result  by various statistics

pre-processing detail:

- consider communities with `> ${threshold}` as "good" ones
- only takes those:
  - with two good ones
  - they do not overlap"
"""

import sys
import pandas as pd
import numpy as np
import networkx as nx
import scipy.io as sio

from helpers import (
        pos_adj, neg_adj
    
)
from stat_helpers import populate_fields

graph = sys.argv[1]

g = nx.read_gpickle('graphs/{}.pkl'.format(graph))
A = nx.adj_matrix(g, weight='sign')
fog_data = sio.loadmat('outputs/focg-{}.mat'.format(graph))

pos_A = pos_adj(A)
neg_A = neg_adj(A)

# for weighted matrix, make it unweighted
if not (pos_A.data == 1).all():
        pos_A.data = np.ones(pos_A.data.shape, dtype='float64')
if not (neg_A.data == 1).all():
        neg_A.data = np.ones(neg_A.data.shape, dtype='float64')

threshold = 5

fog_comms = []
for i in range(fog_data['X_enumKOCG_cell'].shape[0]):
    r = fog_data['X_enumKOCG_cell'][i][0]
    if r.nnz > threshold * 2:
        r1 = r.tocsc()
        nnz_per = [r1[:, i].nnz for i in range(r.shape[1])]
        nnz_per = np.array(nnz_per)
        sorted_sizes = np.sort(nnz_per)[::-1]
        idx = (nnz_per > threshold)
        if sorted_sizes[1] > threshold:
            if idx.sum() == 2:
                fog_comms.append(r[:, idx])

print('got {} entries'.format(len(fog_comms)))

rows = []
for comm in fog_comms:
    C1 = comm[:, 0].nonzero()[0]
    C2 = comm[:, 1].nonzero()[0]
    if not set(C1).intersection(set(C2)):
        row = dict(
            C1=C1,
            C2=C2,
            size1=len(C1),
            size2=len(C2),
        )
        rows.append(row)
df = pd.DataFrame.from_records(rows)
assert df.shape[0] > 0
print('excluding overlapping results, got {} entries'.format(df.shape[0]))

df = populate_fields(df, pos_A, neg_A, make_assertion=False)

df.to_pickle('outputs/focg_{}_aug.pkl'.format(graph))
