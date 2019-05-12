import sys
import numpy as np
import networkx as nx
from helpers import degree_diag, sample_nodes_by_log_of_degree

np.random.seed(12345)

graph_name = sys.argv[1]

graph_path = 'graphs/{}.pkl'.format(graph_name)

if graph_name in ('word', 'bitcoin'):
    n_samples = 1000
    chunk_size = 200
elif graph_name in ('ref', ):
    n_samples = 2000
    chunk_size = 100
elif graph_name in ('slashdot', 'epinions'):
    n_samples = 4000
    chunk_size = 50
elif graph_name in ('wikiconflict'):
    n_samples = 5000
    chunk_size = 40
else:
    raise ValueError('unknown graph', graph_name)

    
kappa = 0.9

cmd = "python3 query_single_seed_in_batch.py -g {} -q {{}} -k {:.1f} -d".format(graph_path, kappa)

g = nx.read_gpickle(graph_path)
D = degree_diag(g)


sys.stderr.write('n_samples={} ({:.2f}% of all nodes in graph)\n'.format(
    n_samples, 100 * n_samples / g.number_of_nodes())
)

sample_queries = sample_nodes_by_log_of_degree(D, n_samples)


n_chunks = int((n_samples + chunk_size - 1) / chunk_size)
sys.stderr.write('n_chunks: {}'.format(n_chunks))
for i in range(n_chunks):
    queries_chunk = sample_queries[i*chunk_size:(i+1)*chunk_size]
    queries_str = ' '.join(map(str, queries_chunk))
    sys.stdout.write(cmd.format(queries_str) + '\n')
