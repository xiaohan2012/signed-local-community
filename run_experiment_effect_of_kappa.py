import numpy as np
import random
import pandas as pd
from joblib import delayed, Parallel
from helpers import sample_seeds, noise_level
from exp_helpers import run_pipeline
from data_helpers import make_polarized_graphs_fewer_parameters


np.random.seed(12345)
random.seed(12345)


def run_one_for_parallel(g, true_comms, true_groupings, kappa, seed_size, nl, run_id):
    seeds, target_comm = sample_seeds(true_comms, true_groupings, k=seed_size)
    try:
        res = run_pipeline(g, seeds, kappa, target_comm, true_comms, true_groupings, verbose=0)
        res['kappa'] = kappa
        res['seed_size'] = seed_size
        res['edge_noise_level'] = nl
        res['run_id'] = run_id
        return res
    except RuntimeError:
        return dict(
            kappa=None,
            seed_size=None,
            edge_noise_level=None,
            run_id=None
        )

n_graphs = 10
n_reps = 60
kappa = 0.8

nc, nn = 10, 0
k = 6
eta = 0.2
seed_size = 9
kappa_list = np.linspace(0.5, 0.95, 10)

if __name__ == "__main__":
    perf_list = []

    for kappa in kappa_list:
        print('kappa', kappa)
        for i in range(n_graphs):
            g, true_comms, true_groupings = make_polarized_graphs_fewer_parameters(
                nc, nn, k, eta, verbose=0
            )
            nl = noise_level(g)
            print('noisy edge ratio: ', nl)
            perf_list += Parallel(n_jobs=8)(
                delayed(run_one_for_parallel)(g, true_comms, true_groupings, kappa, seed_size, nl, i)
                for i in range(n_reps)
            )

    perf_df = pd.DataFrame.from_records(perf_list)
    perf_df.to_csv('outputs/effect_of_kappa.csv')
