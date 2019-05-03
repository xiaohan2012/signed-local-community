import random
import numpy as np
import pandas as pd


from tqdm import tqdm
from scipy.stats import rankdata
from joblib import Parallel, delayed

from core import query_graph
from helpers import noise_level, sample_seeds
from exp_helpers import run_pipeline
from data_helpers import make_polarized_graphs_fewer_parameters


def argmin_kappa_by_seed_rank(g, seeds, kappa_min=0.1, kappa_max=0.95, num_kappa=18):
    mean_rank_list = []
    kappa_list = np.linspace(kappa_min, kappa_max, num_kappa)
    
    for kappa in kappa_list:
        try:
            x_opt, opt_val = query_graph(g, seeds, kappa)
            
            ranks0 = rankdata(x_opt)[seeds[0]]
            ranks1 = rankdata(x_opt)[seeds[1]]
            ranks0 = np.minimum(g.number_of_nodes() - ranks0, ranks0)
            ranks1 = np.minimum(g.number_of_nodes() - ranks1, ranks1)
            mean_rank_list.append(np.mean(np.hstack([ranks0, ranks1])))
        except RuntimeError:
            mean_rank_list.append(float('inf'))
            
    return kappa_list[np.argmin(mean_rank_list)]


def run_one_for_parallel(g, true_comms, true_groupings, seeds, target_comm, kappa, eta, nl, run_id):
    try:
        res = run_pipeline(g, seeds, kappa, target_comm, true_comms, true_groupings, verbose=0)
        
        res['kappa'] = kappa
        res['seed_size'] = len(seeds[0])
        res['run_id'] = run_id
        return res
    except RuntimeError:
        # cg error
        return False


random.seed(12345)
np.random.seed(12345)

DEBUG = False

nc, nn = 10, 0
k = 6
seed_size = 3
eta = 0.1


if DEBUG:
    n_graphs = 1
    n_reps = 6
    seed_size_list = np.arange(1, 2)
else:
    n_graphs = 10
    n_reps = 64
    seed_size_list = np.arange(1, nc)


perf_list = []

constant_kappa_list = [0.1, 0.5, 0.9]

for i in range(n_graphs):
    for seed_size in tqdm(seed_size_list):
        g, true_comms, true_groupings = make_polarized_graphs_fewer_parameters(
            nc, nn, k, eta, verbose=0
        )
        nl = noise_level(g)
        print("seed_size=", seed_size)
        print('|V|, |E|', g.number_of_nodes(), g.number_of_edges())
        print('noisy edge ratio: ', nl)
    
        seed_configs = [sample_seeds(true_comms, true_groupings, k=seed_size) for i in range(n_reps)]

        # run on constant kappas
        for kappa in constant_kappa_list:
            perf_list += Parallel(n_jobs=8)(
                delayed(run_one_for_parallel)(
                    g, true_comms, true_groupings,
                    seeds, target_comm, kappa, eta, nl, "constant{:.1f}".format(kappa)
                )
                for seeds, target_comm in seed_configs
            )

        # run on heuristic kappas
        heuristic_kappas = Parallel(n_jobs=8)(
            delayed(argmin_kappa_by_seed_rank)(
                g, seeds=seeds
            )
            for seeds, target_comm in seed_configs
        )
        
        perf_list += Parallel(n_jobs=8)(
            delayed(run_one_for_parallel)(
                g, true_comms, true_groupings, seeds, target_comm, kappa, eta, nl, "heuristic"
            )
            for kappa, (seeds, target_comm) in zip(heuristic_kappas, seed_configs)
        )

perf_df = pd.DataFrame.from_records(list(filter(None, perf_list)))
perf_df.to_csv('outputs/kappa_heursitic.csv')
