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


def run_one_for_parallel(g, true_comms, true_groupings, kappa, eta, nl, run_id):
    seeds, target_comm = sample_seeds(true_comms, true_groupings)
    res = run_pipeline(
        g, seeds, kappa, target_comm, true_comms, true_groupings,
        verbose=0,
        return_details=True
    )
    
    res['kappa'] = kappa
    res['eta'] = eta
    res['noisy_edge_ratio'] = nl
    res['run_id'] = run_id
    res['ground_truth_beta'] = sbr(
        nx.adj_matrix(g, weight='sign'),
        true_groupings[target_comm][0], true_groupings[target_comm][1]
    )
    return res


DEBUG = False

kappa_list = [0.1, 0.5, 0.7, 0.8, 0.9]
nc, nn = 20, 0
k = 8

if DEBUG:
    n_graphs = 1
    n_reps = 1
    n_jobs = 1
    eta_list = np.linspace(0.01, 0.30, 9)
else:
    n_graphs = 10
    n_reps = 32
    n_jobs = 8
    eta_list = np.linspace(0.01, 0.30, 30)
    
perf_list = []

for eta in tqdm(eta_list):
    for i in range(n_graphs):
        g, true_comms, true_groupings = make_polarized_graphs_fewer_parameters(nc, nn, k, eta, verbose=0)
        nl = noise_level(g)
        print("eta=", eta)
        print('|V|, |E|', g.number_of_nodes(), g.number_of_edges())
        print('noisy edge ratio: ', nl)

        for kappa in kappa_list:
            print('kappa={:.2f}'.format(kappa))
            perf_list += Parallel(n_jobs=n_jobs)(
                delayed(run_one_for_parallel)(g, true_comms, true_groupings, kappa, eta, nl, i)
                for i in range(n_reps)
            )

perf_df = pd.DataFrame.from_records(perf_list)
perf_df.to_csv('outputs/effect_of_eta.csv')
