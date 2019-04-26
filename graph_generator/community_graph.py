import os
import sys
import networkx as nx
import numpy as np
import itertools as it
import pickle as pkl
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from helpers import make_range, round_up


def make_random_signed_graph(N, density=0.9, negatives_ratio=0.5):
    assert density >= 0 and density <= 1, density
    assert negatives_ratio >= 0 and negatives_ratio <= 1, negatives_ratio

    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    for e in it.combinations(range(N), 2):
        if np.random.rand() < density:
            w = 1
            label = 1  # 1 means "good" edge
            if np.random.rand() < negatives_ratio:
                w = -1
                label = 0  # 0 means "noisy" edge
            G.add_edge(e[0], e[1], sign=w, label=label)
    return G


def connect_communities(comm_list, edge_proba=0.3, neg_ratio=0.8, as_noise=False):
    """
    comm_list: list of nx.Graph, each as a community
    edge_proba: probability that an edge, regardless of sign, exists between two communities
    neg_ratio: the ratio of - edge among those existing cut edges
    """
    assert edge_proba >= 0 and edge_proba <= 1
    assert neg_ratio >= 0 and neg_ratio <= 1

    relabeled_comm_list = []
    nodes_by_comm = []
    acc = 0
    for i, c in enumerate(comm_list):
        mapping = {n:  acc+i for i, n in enumerate(c.nodes())}
        new_comm = nx.relabel_nodes(c, mapping)
        relabeled_comm_list.append(new_comm)
        nodes_by_comm.append(list(mapping.values()))
        acc += new_comm.number_of_nodes()
    
    new_g = nx.compose_all(relabeled_comm_list)
        
    # now connect them
    for c1, c2 in it.combinations(nodes_by_comm, 2):
        for u, v in it.product(c1, c2):
            if np.random.rand() < edge_proba:
                if np.random.random() < neg_ratio:
                    sign = - 1.0
                    label = (0 if as_noise else 1)
                else:
                    sign = 1.0
                    label = 0
                new_g.add_edge(u, v, sign=sign, label=label)
    return nx.convert_node_labels_to_integers(new_g), nodes_by_comm


def make(
    n, k, internal_density, internal_negative_ratio, external_edge_proba, external_neg_ratio
):
    """
    example:
    
    >> from helpers import signed_layout, draw_nodes, draw_edges
    >> g = make_community_graph(10, 4, 0.8, 0.0, 0.2, 0.9)
    >> pos = signed_layout(g)
    >> draw_nodes(g, pos)
    >> draw_edges(g, pos)
    """
    comms = [make_random_signed_graph(n, internal_density, internal_negative_ratio)
             for i in range(k)]
    g, nodes_by_comm = connect_communities(comms, external_edge_proba, external_neg_ratio)
    return g, nodes_by_comm


def makedir_if_not_there(d):
    if not os.path.exists(d):
        os.makedirs(d)
            

def make_and_dump_batch(
        n_rep,
        n, k,
        internal_density_list,
        internal_negative_ratio_list,
        external_edge_proba_list,
        external_neg_ratio_list,
        output_dir
):
    params = [
        internal_density_list,
        internal_negative_ratio_list,
        external_edge_proba_list,
        external_neg_ratio_list
    ]
    for p1, p2, p3, p4 in tqdm(
            it.product(*params),
            total=np.prod(list(map(len, params)))
    ):
        rounded_values = list(map(lambda v: round_up(v, 1), [p1, p2, p3, p4]))
        p1, p2, p3, p4 = rounded_values
        outer_dir = "{}/id{:.1f}-in{:.1f}-ep{:.1f}-en{:.1f}".format(
            output_dir, *rounded_values
        )
        makedir_if_not_there(outer_dir)

        for i in range(n_rep):
            params = dict(
                n=n, k=k,
                internal_density=p1,
                internal_negative_ratio=p2,
                external_edge_proba=p3,
                external_neg_ratio=p4
            )

            g, comm = make(**params)
            output_path = '{}/{}.pkl'.format(outer_dir, i)
            
            pkl.dump((g, comm, params), open(output_path, 'wb'))

  
if __name__ == '__main__':
    internal_density_list = make_range(0.5, 1.0)
    internal_negative_ratio_list = make_range(0, 0.5)
    external_edge_proba_list = make_range(0.0, 0.5)
    external_neg_ratio_list = make_range(0.6, 1.0)

    n_reps = 10
    n, k = 16, 4
    root_dir = "/scratch/work/xiaoh1/data/signed-local-community"
    output_dir = "{}/community_graphs-n{}-k{}".format(root_dir, n, k)
    make_and_dump_batch(
        n_reps,
        n, k,
        internal_density_list,
        internal_negative_ratio_list,
        external_edge_proba_list,
        external_neg_ratio_list,
        output_dir=output_dir
    )

