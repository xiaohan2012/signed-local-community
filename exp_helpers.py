import numpy as np
import scipy
import networkx as nx
from matplotlib import pyplot as plt
from core import query_graph_using_dense_matrix, sweep_on_x
from eval_helpers import evaluate_level_1, evaluate_level_2
from helpers import (
    flatten,
    get_borderless_fig,
    signed_layout,
    draw_edges,
    sbr
)    


def run_pipeline(
        g,
        seeds,
        kappa,
        target_comm,
        true_comms,
        true_groupings,
        # debugging swtiches
        check_bound=True,  # check if the approximation ratio holds
        show_sweep_plot=False,  # show the sweep plot
        plot_returned_subgraph=False,  # plot the returned subgraph
        plot_true_community=False,  # plot the true p--community
        verbose=0
):
    x_opt, opt_val = query_graph_using_dense_matrix(g, seeds, kappa=kappa, verbose=0)
    c1, c2, C, min_sbr, ts, sbr_list = sweep_on_x(g, x_opt, verbose=0)

    prec_L1, rec_L1, f1_L1 = evaluate_level_1(
        g.number_of_nodes(), C, true_comms[target_comm]
    )[:-1]
    prec_L2, rec_L2, f1_L2 = evaluate_level_2(
        g.number_of_nodes(), c1, c2, C, true_groupings[target_comm]
    )[:-1]

    # debugging stuff below
    if verbose > 1:
        print('nodes ordered by x_opt (asc):')
        print(np.argsort(x_opt))
        print('comm1: ', np.sort(c1))
        print('comm2: ', np.sort(c2))
        print('true comm1: ', true_groupings[target_comm][0])
        print('true comm2: ', true_groupings[target_comm][1])

    if show_sweep_plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(ts, sbr_list)

        best_t = ts[np.argmin(sbr_list)]
        ax.axvline(best_t, color='red')
        ax.set_xlabel('threshold')
        ax.set_ylabel('beta')
        ax.set_title('sweeping profile')

    if plot_returned_subgraph:
        subg = g.subgraph(C)
        fig, ax = get_borderless_fig()
        subg = nx.convert_node_labels_to_integers(subg)
        new_pos = signed_layout(subg)
        nx.draw_networkx_nodes(subg, new_pos, node_size=100)
        # draw_nodes(subg, new_pos, ax=ax)
        draw_edges(subg, new_pos, ax=ax)
        ax.set_title('predicted subgraph')

    if plot_true_community:
        A = nx.adj_matrix(g, weight='sign')
        true_comm = true_comms[target_comm]
        # relevant_nodes = the community | adjacent nodes
        relevant_nodes = scipy.absolute(A[true_comm, :]).sum(axis=0).nonzero()[1]

        subg = g.subgraph(relevant_nodes)
        mapping = {n: i for i, n in enumerate(relevant_nodes)}
        subg = nx.relabel_nodes(subg, mapping=mapping)

        color = np.zeros(subg.number_of_nodes())
        color[[mapping[i] for i in true_comm]] = 1
        new_pos = signed_layout(subg)
        fig, ax = get_borderless_fig()
        nx.draw_networkx_nodes(
            subg, new_pos, node_size=50, node_color=color, ax=ax, cmap=plt.cm.coolwarm
        )
        # draw_edges(subg, pos=new_pos, ax=ax)
        ax.set_title('true community (red) with adjacent nodes (blue)')
        
    if check_bound:
        # check the bound according to Proposition 1
        print('beta=', min_sbr)
        print('upperbound sqrt(opt_val)=', np.sqrt(2 * opt_val))
        does_hold = min_sbr <= np.sqrt(2 * opt_val)
        assert does_hold
        print('does upperbound hold?', does_hold)
        print('-' * 10)

    if verbose > 0:
        A = nx.adj_matrix(g, weight='sign')
        true_c1, true_c2 = true_groupings[target_comm]

        def show_community_stats(V1, V2):
            beta_val, details = sbr(A, V1, V2, return_details=True)
            print('beta_val=', beta_val)
            print('beta calcualte details:', details)

        print('-' * 10)
        print('stats of ground truth p-community:')
        print('community size=', len(true_c1) + len(true_c2))
        show_community_stats(true_c1, true_c2)

        print('-' * 10)
        print('stats of predicted p-community:')
        print('predicted community size=', len(c1) + len(c2))
        show_community_stats(c1, c2)
        
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
        C1=c1,
        C2=c2,
        min_beta=min_sbr,
        seeds=list(flatten(seeds))
    )
