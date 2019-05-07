import sys
import numpy as np
import networkx as nx

np.random.seed(12345)

graph_path = 'graphs/slashdot1.pkl'
kappa = 0.9

cmd = "python3 run_queries_in_batch.py -g {} -q {{}} -k {:.1f} -d".format(graph_path, kappa)

g = nx.read_gpickle(graph_path)

n_samples = 8000
chunk_size = 50

sys.stderr.write('n_samples={} ({:.2f}% of all nodes in graph)\n'.format(
    n_samples, 100 * n_samples / g.number_of_nodes())
)

sample_queries = np.random.permutation(g.number_of_nodes())[:n_samples]



n_chunks = int((n_samples + chunk_size - 1) / chunk_size)
sys.stderr.write('n_chunks: {}'.format(n_chunks))
for i in range(n_chunks):
    queries_chunk = sample_queries[i*chunk_size:(i+1)*chunk_size]
    queries_str = ' '.join(map(str, queries_chunk))
    sys.stdout.write(cmd.format(queries_str) + '\n')
