import numpy as np
from core import query_graph_using_dense_matrix, sweep_on_x
from eval_helpers import evaluate_level_1, evaluate_level_2
from helpers import flatten


def run_pipeline(
        g, seeds, kappa, target_comm, true_comms, true_groupings, check_bound=True
):
    x_opt, opt_val = query_graph_using_dense_matrix(g, seeds, kappa=kappa, verbose=0)
    c1, c2, C, min_sbr = sweep_on_x(g, x_opt, verbose=0)
    prec_L1, rec_L1, f1_L1 = evaluate_level_1(
        g.number_of_nodes(), C, true_comms[target_comm]
    )[:-1]
    prec_L2, rec_L2, f1_L2 = evaluate_level_2(
        g.number_of_nodes(), c1, c2, C, true_groupings[target_comm]
    )[:-1]

    if check_bound:
        # check the bound according to Proposition 1
        print('beta=', min_sbr)
        print('upperbound sqrt(opt_val)=', np.sqrt(2 * opt_val))
        does_hold = min_sbr <= np.sqrt(2 * opt_val)
        assert does_hold
        print('does upperbound hold?', does_hold)
        print('-' * 10)
    
    return dict(
        prec_L1=prec_L1,
        rec_L1=rec_L1,
        f1_L1=f1_L1,
        prec_L2=prec_L2,
        rec_L2=rec_L2,
        f1_L2=f1_L2,
        C_size=len(C),
        C1_size=len(c1),
        C2_size=len(c2),
        min_beta=min_sbr,
        seeds=list(flatten(seeds))
    )

