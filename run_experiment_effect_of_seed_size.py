import numpy as np
import random
import pandas as pd
import networkx as nx

from helpers import sample_seeds, noise_level, get_theoretical_kappa, sbr
from data_helpers import make_polarized_graphs_fewer_parameters
from exp_helpers import run_pipeline
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(12345)
random.seed(12345)


def run_one_for_parallel(g, true_comms, true_groupings, kappa_info, seed_size, nl, run_id):
    seeds, target_comm = sample_seeds(true_comms, true_groupings, k=seed_size)
    try:
        if kappa_info['use_suggested_kappa']:
            kappa = get_theoretical_kappa(
                true_comms[target_comm],
                seeds,
                nx.adj_matrix(g, weight='sign')
            )
        else:
            kappa = kappa_info['kappa_default']

        res = run_pipeline(g, seeds, kappa, target_comm, true_comms, true_groupings, verbose=0)
        res['kappa'] = kappa
        res['seed_size'] = seed_size
        res['edge_noise_level'] = nl
        res['run_id'] = run_id
        res['ground_truth'] = true_groupings[target_comm]
        res['ground_truth_beta'] = sbr(
            nx.adj_matrix(g, weight='sign'),
            true_groupings[target_comm][0], true_groupings[target_comm][1]
        )
        return res
    except RuntimeError:
        return dict(
            kappa=None,
            seed_size=None,
            edge_noise_level=None,
            run_id=None
        )

kappa_info = dict(
    use_suggested_kappa=False,
    kappa_default=0.8
)


DEBUG = False

nc, nn = 10, 0
k = 6
eta = 0.2

if DEBUG:
    n_graphs = 1
    n_reps = 8
    seed_size_list = np.arange(1, 3)
else:
    n_graphs = 10
    n_reps = 60
    seed_size_list = np.arange(1, nc)


if __name__ == "__main__":
    perf_list = []

    for seed_size in tqdm(seed_size_list):
        for i in range(n_graphs):
            g, true_comms, true_groupings = make_polarized_graphs_fewer_parameters(
                nc, nn, k, eta, verbose=0
            )
            nl = noise_level(g)
            print("seed_size=", seed_size)
            print('noisy edge ratio: ', nl)
            
            perf_list += Parallel(n_jobs=8)(
                delayed(run_one_for_parallel)(g, true_comms, true_groupings, kappa_info, seed_size, nl, i)
                for i in range(n_reps)
            )

    perf_df = pd.DataFrame.from_records(perf_list)

    if kappa_info['use_suggested_kappa']:
        output_path = 'outputs/effect_of_seed_size_suggested_kappa.csv'
    else:
        output_path = 'outputs/effect_of_seed_size.csv'
    print('saving to', output_path)
    perf_df.to_csv(output_path)
