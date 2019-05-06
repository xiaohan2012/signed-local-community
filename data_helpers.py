import numpy as np
import networkx as nx
from graph_generator.community_graph import (
    make_random_signed_graph,
    connect_communities
)
from itertools import combinations
from helpers import num_good_edges, num_bad_edges


def make_polarized_graphs(
        k, comm_sizes,
        internal_density=0.9,
        internal_neg_ratio=0.05,
        comm_cross_edge_proba=0.5,
        comm_cross_neg_ratio=0.95,
        cross_edge_proba=0.01,
        cross_neg_ratio=0.5,
        verbose=0
):
    """
    generate k pairs of polarized communities,
    where the size is determined by comm_sizes of form:
    [
      (size of 1st community of pair 1, size of 2nd community of pair 1),
      (size of 1st community of pair 2, size of 2nd community of pair 2),
      ...
    ]
    under possibly noisy setting
    """
    if verbose > 0:
        print('#communities', k)
    assert k == len(comm_sizes)
    comms = []
    size_acc = 0
    groupings = []
    for i, sizes in zip(range(k), comm_sizes):
        assert len(sizes) == 2
        n1, n2 = sizes
        c0 = make_random_signed_graph(n1, internal_density, internal_neg_ratio)
        c1 = make_random_signed_graph(n2, internal_density, internal_neg_ratio)
        c, groups = connect_communities(
            [c0, c1],
            edge_proba=comm_cross_edge_proba,
            neg_ratio=comm_cross_neg_ratio
        )
        if verbose > 0:
            print('comm#{} sizes: {} {}'.format(i+1, c0.number_of_nodes(), c1.number_of_nodes()))
            if verbose > 1:
                print('num. good edges', num_good_edges(c))
                print('num. bad edges', num_bad_edges(c))
        groups = np.asarray(groups)
        groups += size_acc
        groupings.append(groups.tolist())
        comms.append(c)
        size_acc += c.number_of_nodes()

    # connect the k pairs
    # edges in between are **all noise**
    g, comms = connect_communities(
        comms, edge_proba=cross_edge_proba, neg_ratio=cross_neg_ratio,
        as_noise=True
    )

    return g, comms, groupings


def make_polarized_graphs_fewer_parameters(
        nc, nn, k, eta, verbose=0
):
    """
    nc: size of polarized community or list of community size pairs
    nn: number of irrelevant numbers
    k: number of polarized community pairs
    eta: noise level
    
    edge generation process:
    
    - edges inside S1 (respect. S2) exist and are positive with probability
    1 − η, exist and are negative with probability η/2, and
    do not exist with probability η/2;
    
    - edges between S1 and S2 exist and are negative with probability
    1 − η, exist and are positive with probability η/2, and do not
    exist with probability η/2;
    
    - all other edges (outside the two polarized communities) exist
    with probability η and have equal probability of being positive
    or negative.
    """
    def _aux():
        assert eta >= 0
        assert eta <= 1

        if isinstance(nc, int):
            comm_sizes = [(nc, nc) for i in range(k)]
        else:
            assert isinstance(nc, list)
            for pair in nc:
                assert len(pair) == 2

            comm_sizes = nc
            
        inside_edge_proba = 1-eta/2
        ind = inside_edge_proba
        inr = (eta / 2) / ind
        ccep = inside_edge_proba
        ccnr = (1-eta)/ccep

        if verbose > 0:
            print('internal_density', ind)
            print('internal_neg_ratio', inr)

        g, comms, groupings = make_polarized_graphs(
            k, comm_sizes,
            internal_density=ind,
            internal_neg_ratio=inr,
            comm_cross_edge_proba=ccep,
            comm_cross_neg_ratio=ccnr,
            cross_edge_proba=eta,
            cross_neg_ratio=0.5,
            verbose=verbose
        )
            
        if verbose > 0:
            # print edge statistics (by whether it's noisy or not)
            edge_labels_inside = np.array([
                g[u][v]['label'] for comm in comms for u, v in g.subgraph(comm).edges()
            ])
            edge_labels = np.array([
                g[u][v]['label'] for u, v in g.edges()
            ])
            num_good_edges = edge_labels.sum()
            num_noisy_edges = edge_labels[edge_labels == 0].shape[0]
            num_noisy_edges_inside_community = edge_labels_inside[edge_labels_inside == 0].shape[0]
            num_noisy_edges_among_communities = (
                num_noisy_edges - num_noisy_edges_inside_community
            )
            assert (num_good_edges + num_noisy_edges) == g.number_of_edges()
            print('-' * 15)
            print('num. good edges=', num_good_edges)
            print('num. noisy edges inside pairs=', num_noisy_edges_inside_community)
            print('num. noisy edges among pairs=', num_noisy_edges_among_communities)
       
        cur_n = g.number_of_nodes()
        comm_nodes = np.arange(cur_n)

        # add noisy nodes and edges
        noisy_nodes = list(range(cur_n, cur_n + nn))

        def add_noisy_edge_randomly(u, v):
            if np.random.rand() < eta:
                if np.random.rand() >= 0.5:
                    g.add_edge(u, v, sign=1, label=0)
                else:
                    g.add_edge(u, v, sign=-1, label=0)

        g.add_nodes_from(noisy_nodes)
        for u, v in combinations(noisy_nodes, 2):
            add_noisy_edge_randomly(u, v)

        for u in comm_nodes:
            for v in noisy_nodes:
                add_noisy_edge_randomly(u, v)
        
        return g, comms, groupings

    while True:
        g, comms, groupings = _aux()
        if len(list(nx.connected_components(g))) == 1:
            break
        print('gen_graph: not connected, repeat')

    return g, comms, groupings
