import numpy as np
import scipy
import networkx as nx
import warnings

from matplotlib import pyplot as plt
from core import query_graph, sweep_on_x_fast
from eval_helpers import mean_avg_precision
from helpers import (
    flatten,
    get_borderless_fig,
    signed_layout,
    draw_edges,
    sbr
)


def warn(*args, **kwargs):
        pass

warnings.warn = warn


def run_pipeline(
        g,
        seeds,
        kappa,
        target_comm,
        true_comms,
        true_groupings,
        max_iter=40,
        tol=1e-3,
        # debugging swtiches
        check_bound=True,  # check if the approximation ratio holds
        show_sweep_plot=False,  # show the sweep plot
        plot_returned_subgraph=False,  # plot the returned subgraph
        plot_true_community=False,  # plot the true p--community
        return_details=False,
        verbose=0
):
    # x_opt, opt_val = query_graph_using_dense_matrix(g, seeds, kappa=kappa, verbose=verbose)
    x_opt, opt_val, details = query_graph(
            g, seeds, kappa=kappa, verbose=verbose, solver='cg',
            return_details=return_details,
            max_iter=max_iter,
            tol=tol
    )
    c1, c2, C, best_t, min_sbr, ts, sbr_list = sweep_on_x_fast(g, x_opt, verbose=verbose)

    map_score = mean_avg_precision(g, c1, c2, target_comm, true_groupings)
    assert map_score >= 0 and map_score <= 1

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
        does_hold = min_sbr <= np.sqrt(2 * opt_val)
        if verbose > 0:
            print('beta=', min_sbr)
            print('upperbound sqrt(opt_val)=', np.sqrt(2 * opt_val))
            print('does upperbound hold?', does_hold)
            print('-' * 10)

        assert does_hold

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

    deg = flatten(nx.adjacency_matrix(g).sum(axis=0))

    return dict(
        MAP=map_score,
        C_size=len(C),
        C1_size=len(c1),
        C2_size=len(c2),
        C1_vol=deg[c1].sum(),
        C2_vol=deg[c2].sum(),
        C1=c1,
        C2=c2,
        min_beta=min_sbr,
        seeds=list(flatten(seeds)),
        kappa=kappa,
        runtime_details=(return_details and details or None)
    )
