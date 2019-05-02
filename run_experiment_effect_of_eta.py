import numpy as np
import random
import pandas as pd

from helpers import sample_seeds, noise_level
from data_helpers import make_polarized_graphs_fewer_parameters
from exp_helpers import run_pipeline
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(12345)
random.seed(12345)


def run_one_for_parallel(g, true_comms, true_groupings, kappa, eta, nl, run_id):
    seeds, target_comm = sample_seeds(true_comms, true_groupings)
    res = run_pipeline(g, seeds, kappa, target_comm, true_comms, true_groupings, verbose=0)

    res['kappa'] = kappa
    res['eta'] = eta
    res['edge_noise_level'] = nl
    res['run_id'] = run_id
    return res


n_graphs = 10
n_reps = 60
kappa = 0.8

nc, nn = 10, 0
k = 6

head_eta_list = np.linspace(0.01, 0.30, 9)
perf_list = []

for eta in tqdm(head_eta_list):
    for i in range(n_graphs):
        g, true_comms, true_groupings = make_polarized_graphs_fewer_parameters(nc, nn, k, eta, verbose=0)
        nl = noise_level(g)
        print("eta=", eta)
        print('|V|, |E|', g.number_of_nodes(), g.number_of_edges())
        print('noisy edge ratio: ', nl)

        perf_list += Parallel(n_jobs=8)(
            delayed(run_one_for_parallel)(g, true_comms, true_groupings, kappa, eta, nl, i)
            for i in range(n_reps)
        )

perf_df = pd.DataFrame.from_records(perf_list)
perf_df.to_csv('outputs/result_synthetic_graphs_cg.csv')
