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


def run_one_for_parallel(g, true_comms, true_groupings, kappa, nn, nl, run_id):
    try:
        seeds, target_comm = sample_seeds(true_comms, true_groupings)
        res = run_pipeline(g, seeds, kappa, target_comm, true_comms, true_groupings, verbose=0, return_details=True)
        res['ground_truth_beta'] = sbr(
            nx.adj_matrix(g, weight='sign'),
            true_groupings[target_comm][0], true_groupings[target_comm][1]
        )
    except AssertionError as e:
        import pickle as pkl
        print('dumping result')
        pkl.dump(
            dict(
                g=g, true_comms=true_comms, true_groupings=true_groupings,
                kappa=kappa, nn=nn, nl=nl, run_id=run_id,
                seeds=seeds, target_comm=target_comm
            ),
            open('dumps/scene.pkl', 'wb')
        )
        print(e)
        raise e

    res['kappa'] = kappa
    res['nn'] = nn
    res['edge_noise_level'] = nl
    res['run_id'] = run_id
    return res


DEBUG = False

if DEBUG:
    n_graphs = 1
    n_reps = 1
else:
    n_graphs = 10
    n_reps = 32

nc = 20
k = 8
eta = 0.03

nn_list = np.arange(0, nc*2*k + 1, 2*nc)
kappa_list = [0.1, 0.3, 0.5, 0.7, 0.9]

perf_list = []

for nn in tqdm(nn_list):
    for i in range(n_graphs):
        g, true_comms, true_groupings = make_polarized_graphs_fewer_parameters(nc, nn, k, eta, verbose=0)
        nl = noise_level(g)
        print("nn=", nn)
        print('noisy edge ratio: ', nl)
        for kappa in kappa_list:
            print("kappa=", kappa)
            perf_list += Parallel(n_jobs=8)(
                delayed(run_one_for_parallel)(g, true_comms, true_groupings, kappa, nn, nl, i)
                for i in range(n_reps)
            )

perf_df = pd.DataFrame.from_records(perf_list)
perf_df.to_csv('outputs/effect_of_outlier_size.csv')
