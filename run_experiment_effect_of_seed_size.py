import numpy as np
import random
import pandas as pd
import networkx as nx

from helpers import sample_seeds, noise_level, sbr
from data_helpers import make_polarized_graphs_fewer_parameters
from exp_helpers import run_pipeline
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(12345)
random.seed(12345)


def run_one_for_parallel(g, true_comms, true_groupings, kappa, seed_size, nl, run_id):
    seeds, target_comm = sample_seeds(true_comms, true_groupings, k=seed_size)
    res = run_pipeline(
        g, seeds, kappa, target_comm, true_comms, true_groupings,
        max_iter=40,
        tol=1e-3,
        verbose=0, return_details=True
    )
    res['kappa'] = kappa
    res['seed_size'] = seed_size
    res['edge_noise_level'] = nl
    res['run_id'] = run_id
    res['alpha'] = res['runtime_details']['alpha']
    res['lambda1'] = float(np.real(res['runtime_details']['lambda1']))
    res['converged'] = res['runtime_details']['converged']

    del res['runtime_details']
    res['ground_truth'] = true_groupings[target_comm]
    res['ground_truth_beta'] = sbr(
        nx.adj_matrix(g, weight='sign'),
        true_groupings[target_comm][0], true_groupings[target_comm][1]
    )
    return res

DEBUG = False

nc, nn = 20, 0
k = 8
eta = 0.05

if DEBUG:
    n_graphs = 1
    n_reps = 8
    seed_size_list = np.arange(1, 3)
else:
    # n_graphs = 10
    # n_reps = 60
    n_graphs = 8
    n_reps = 8
    seed_size_list = np.arange(1, 11)

kappa_list = [0.1, 0.5, 0.7, 0.8, 0.9]

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

            for kappa in kappa_list:
                print('kappa =', kappa)
                perf_list += Parallel(n_jobs=8)(
                    delayed(run_one_for_parallel)(g, true_comms, true_groupings, kappa, seed_size, nl, i)
                    for i in range(n_reps)
                )

    perf_df = pd.DataFrame.from_records(perf_list)

    output_path = 'outputs/effect_of_seed_size{}.csv'.format(DEBUG and "_dbg" or "")
    print('saving to', output_path)
    perf_df.to_csv(output_path)
