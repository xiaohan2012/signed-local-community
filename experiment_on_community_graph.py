

import os
import time
import sys
import argparse
import pickle as pkl
from collections import OrderedDict

from parser_helper import (
    add_community_graph_args,
    add_pagerank_args,
    add_detection_methods_args
)
from const import DATA_DIR, DetectionMethods
from algorithms import (
    get_comunity_using_pos_pagerank,
    get_community_by_sweeping_on_true_community,
)
from helpers import evaluate_performance
from sql import TableCreation, init_db, insert_record, record_exists


def get_graph_path(args):
    path = os.path.join(
        DATA_DIR,
        "community_graphs-n{}-k{}".format(args.community_size, args.num_communities),
        "id{:.1f}-in{:.1f}-ep{:.1f}-en{:.1f}".format(
            args.internal_density,
            args.internal_negative_ratio,
            args.external_edge_proba,
            args.external_neg_ratio,
        ),
        "{}.pkl".format(args.graph_id)
    )
    return path


def load_community_graph_data(args):
    path = get_graph_path(args)
    return pkl.load(open(path, 'rb'))


def main():
    parser = argparse.ArgumentParser()
    add_community_graph_args(parser)
    add_pagerank_args(parser)
    add_detection_methods_args(parser)
    
    args = parser.parse_args()
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('-' * 25)

    conn, cursor = init_db()

    # if runs already, exit
    filter_value = dict(
        graph_path=get_graph_path(args),
        method=args.method,
        teleport_alpha=args.teleport_alpha,
        query_node=args.query_node
        
    )
    if record_exists(cursor, TableCreation.comm_graph_exp_table, filter_value):
        print('record exists, exit')
        sys.exit(0)
    
    g, ground_truth, graph_params = load_community_graph_data(args)
    print("load graph done")

    # run detection alforithm
    stime = time.time()
    if args.method == DetectionMethods.SWEEP_ON_TRUE:
        pred_comm = get_community_by_sweeping_on_true_community(g, args.query_node, ground_truth)
    elif args.method == DetectionMethods.PR_ON_POS:
        pred_comm = get_comunity_using_pos_pagerank(g, args.query_node, args.teleport_alpha)

    time_elapsed = time.time() - stime
    print("community detection done")
    
    # evaluation
    pred_comm = set(pred_comm)
    true_comm = set(next(c for c in ground_truth if args.query_node in c))
    perf = evaluate_performance(g, pred_comm, true_comm)

    ans = OrderedDict()
    ans['graph_path'] = get_graph_path(args)
    ans['graph_params'] = graph_params

    ans['method'] = args.method
    ans['query_node'] = args.query_node
    ans['teleport_alpha'] = args.teleport_alpha

    ans['true_comm'] = true_comm
    ans['pred_comm'] = pred_comm

    ans.update(perf)

    ans['time_elapsed'] = time_elapsed
    
    print("answer:")
    print(ans)

    # save result to db
    insert_record(
        cursor, TableCreation.schema, TableCreation.comm_graph_exp_table, ans
    )
    conn.commit()
    print('inserted to db')
    conn.close()
    
if __name__ == '__main__':
    main()

