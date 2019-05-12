import sys
import random
import pickle as pkl
import numpy as np
import networkx as nx
from helpers import degree_diag

np.random.seed(12345)

graph_name = sys.argv[1]

graph_path = 'graphs/{}.pkl'.format(graph_name)
pairs_path = 'outputs/{}_good_pairs.pkl'.format(graph_name)

graph2config = dict(
    word=(4000, 200),
    bitcoin=(1300, 200),
    ref=(5000, 300),
    epinions=(10000, 150),
    slashdot=(10000, 200),
    wikiconflict=(10000, 50)
)

n_samples, chunk_size = graph2config[graph_name]
    
kappa = 0.9

cmd = "python3 query_seed_pair_in_batch.py -g {} --seed1_list {{}} --seed2_list {{}} -k {:.1f} -d".format(graph_path, kappa)

g = nx.read_gpickle(graph_path)

pairs = pkl.load(open(pairs_path, 'rb'))

sys.stderr.write('n_samples={} ({:.2f}% of good pairs)\n'.format(
    n_samples, 100 * n_samples / len(pairs))
)

sampled_seed_pairs = random.sample(pairs, n_samples)

n_chunks = int((n_samples + chunk_size - 1) / chunk_size)
sys.stderr.write('n_chunks: {}'.format(n_chunks))

for i in range(n_chunks):
    queries_chunk = sampled_seed_pairs[i*chunk_size:(i+1)*chunk_size]
    seeds1, seeds2 = zip(*queries_chunk)
    seeds1_str = ' '.join(map(str, seeds1))
    seeds2_str = ' '.join(map(str, seeds2))
    sys.stdout.write(cmd.format(seeds1_str, seeds2_str) + '\n')
