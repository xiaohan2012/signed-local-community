"""
query local community using motif clustering framework
"""

import argparse
import time

from collections import OrderedDict
from functools import reduce

import networkx as nx
import numpy as np

from scipy import sparse as sp
from helpers import (
    motif_primary_key_value,
    node2connected_component
) 
from sweeping import sweeping_scores_using_ppr
from motif_adjacency import MOTIF2F
from parser_helper import (
    add_real_graph_args,
    add_pagerank_args,
    add_motif_args,
    add_misc_args
)
from sql import TableCreation, init_db, insert_record, record_exists
from eval_helpers import community_summary


def find_local_cluster(g, motif_ids, query, alpha):
    A = nx.adj_matrix(g, weight='sign')

    W = reduce(
        lambda a, b: a + b,
        [MOTIF2F[m](A) for m in motif_ids],
        sp.dok_matrix(A.shape).tocsr()
    )    
        
    g_motif = nx.from_scipy_sparse_matrix(W)

    cc_list = list(nx.connected_components(g_motif))
    n2cc = node2connected_component(cc_list)

    cc_nodes = cc_list[n2cc[query]]
    motif_cc = g_motif.subgraph(cc_nodes)

    g2cc_map = {n: i for i, n in enumerate(cc_nodes)}
    cc2g_map = {i: n for i, n in enumerate(cc_nodes)}
    motif_cc = nx.relabel_nodes(motif_cc, mapping=g2cc_map)

    cc_query = g2cc_map[query]

    if motif_cc.degree(cc_query) == 0:
        raise ValueError('query {} has degree 0 in motif graph'.format(query))

    motif_A = nx.adjacency_matrix(motif_cc, weight='weight')
    # print('get node score (local) w.r.t node {}'.format(query))
    order, _, sweep_scores = sweeping_scores_using_ppr(
        motif_cc, cc_query, alpha, A=motif_A
    )

    # get the best community
    best_pos = np.argmin(sweep_scores)
    cc_comm = order[:best_pos+1]
    print('best position', best_pos+1)

    # map back the lcc nodes to g
    comm = [cc2g_map[n] for n in cc_comm]
    return list(sorted(comm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_real_graph_args(parser)
    add_motif_args(parser)
    add_pagerank_args(parser)
    add_misc_args(parser)

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('-' * 25)

    motif_ids = list(sorted(args.motifs))
    assert len(motif_ids) > 0, 'no motifs given'
    for m in motif_ids:
        assert m in MOTIF2F, '{} is not a valid motif id'.format(m)
    graph_path = args.graph_path
    query = args.query_node
    alpha = args.teleport_alpha

    conn, cursor = init_db()

    experiment_id = motif_primary_key_value(graph_path, motif_ids, alpha, query)
    filter_value = dict(
        id=experiment_id
    )

    if record_exists(cursor, TableCreation.query_result_table, filter_value):
        print('record exists, skip')
    else:
        g = nx.read_gpickle(graph_path)

        stime = time.time()
        community = find_local_cluster(g, motif_ids, query, alpha)
        time_elapsed = time.time() - stime

        method_name = 'motif-{}'.format(''.join(motif_ids))
        ans = OrderedDict()
        ans['id'] = experiment_id
        ans['graph_path'] = graph_path
        ans['method'] = method_name
        ans['query_node'] = query
        ans['teleport_alpha'] = alpha
        ans['other_params'] = {}
        ans['community'] = community
        ans['time_elapsed'] = time_elapsed

        print(ans)

        summary = community_summary(g.subgraph(community), g)
        print('summary', summary)

        if args.save_db:
            # community result
            insert_record(
                cursor, TableCreation.schema, TableCreation.query_result_table, ans
            )

            # evaluation result
            eval_ans = dict(
                id=experiment_id,
                graph_path=graph_path,
                method=method_name,
                query_node=query,
                teleport_alpha=alpha,
                other_params={}
            )
            for k, v in summary.items():
                eval_ans['key'] = k
                eval_ans['value'] = float(v)
                insert_record(
                    cursor, TableCreation.schema, TableCreation.eval_result_table, eval_ans
                )
            conn.commit()
            print('inserted to db')
    conn.close()    
