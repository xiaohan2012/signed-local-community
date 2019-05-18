"""
augment a graph such that..

- it has 1M nodes
- its avg degree and negtive edge ratio are preserved w.r.t the original graph
"""

import sys
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from time import time
from tqdm import tqdm

from core import query_graph_using_sparse_linear_solver, sweep_on_x_fast
from helpers import (
    flatten, signed_normalized_laplacian,
    degree_diag, pos_adj, signed_laplacian
)

graph = sys.argv[1]
target_size = int(sys.argv[2])
n_reps = int(sys.argv[3])

print('target_size: ', target_size)
print('n_reps: ', n_reps)

g = nx.read_gpickle('graphs/{}.pkl'.format(graph))

n = g.number_of_nodes()
m = g.number_of_edges()
num_neg_edges = sum((g[u][v]['sign'] < 0) for u, v in g.edges())
neg_frac = num_neg_edges / m

print('neg_frac', neg_frac)


avg_degree = 2*m / n
print('avg_degree', avg_degree)
avg_degree = int(avg_degree)

nodes_to_add = target_size - n
print('nodes to add', nodes_to_add)


new_nodes = np.arange(n, target_size, dtype=int)
all_nodes = np.arange(target_size, dtype=int)

A = nx.adj_matrix(g, weight='sign')

row_idx, col_idx = A.nonzero()
data = flatten(A[row_idx, col_idx])

new_row_idx = np.repeat(new_nodes, int(avg_degree), axis=0)
n_new_edges = int(avg_degree)*len(new_nodes)
new_col_idx = np.random.choice(all_nodes, size=n_new_edges)
new_data = np.ones(n_new_edges, dtype=int)
new_data[np.random.random(n_new_edges) < neg_frac] = -1


final_row_idx = np.concatenate([row_idx, new_row_idx, new_col_idx])
final_col_idx = np.concatenate([col_idx, new_col_idx, new_row_idx])
final_data = np.concatenate([data, new_data, new_data])


# now, operate on the new graph

A = sp.csr_matrix((final_data, (final_row_idx, final_col_idx)), shape=(target_size, target_size))
print('num. nodes (augmented graph)', A.shape[0])
print('num. edges (augmented graph)', A.nnz / 2)

deg = abs(A).sum(axis=0)

L = signed_laplacian(g)
A = nx.adj_matrix(g, weight='sign')
D = degree_diag(g)

assert (deg > 0).sum()


pos_A = pos_adj(A)
pos_deg = flatten(pos_A.sum(axis=0))

Ln = signed_normalized_laplacian(A)
eig_val, eig_vec = eigs(Ln, k=1, which='SM')
v1 = np.real(eig_val[0])

print('lambda1', v1)

# select good seed pairs
neg_u, neg_v = (A < 0).nonzero()
rand_idx = np.random.permutation(neg_u.shape[0])
neg_u, neg_v = neg_u[rand_idx], neg_v[rand_idx]
seed_pairs = []

for (u, v) in zip(neg_u, neg_v):
    if len(seed_pairs) >= 100:
        break
    if pos_deg[u] > 50 and pos_deg[v] > 50:
        seed_pairs.append([[u], [v]])

time_cost_list = []
for seeds in tqdm(seed_pairs):
    start_time = time()
    x, _ = query_graph_using_sparse_linear_solver(g, seeds, ub=v1, max_iter=100, L=L, A=A, D=D)
    time_cost_solve = time() - start_time

    start_time = time()
    _ = sweep_on_x_fast(g, x, top_k=200, A=A)
    time_cost_sweeping = time() - start_time
    time_cost_list.append((time_cost_solve, time_cost_sweeping))


df = pd.DataFrame(time_cost_list, columns=['solve', 'sweep'])

print(df.describe())
df.to_pickle('outputs/scalability-{}-{}.pkl'.format(graph, target_size))

