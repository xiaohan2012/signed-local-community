import argparse
import networkx as nx
import time
from collections import OrderedDict

from parser_helper import (
    add_real_graph_args,
    add_pagerank_args,
    add_detection_methods_args,
    add_misc_args
)
from const import DATA_DIR, DetectionMethods
from algorithms import (
    get_comunity_using_pos_pagerank,
    get_community_by_jumping_pagerank,
)
from helpers import signed_conductance
from sql import TableCreation, init_db, insert_record, record_exists

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    add_real_graph_args(parser)
    add_detection_methods_args(parser)
    add_pagerank_args(parser)
    add_misc_args(parser)
    
    args = parser.parse_args()
        
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('-' * 25)

    conn, cursor = init_db()

    if args.verbose > 0:
        print('reading graph')
    g = nx.read_gpickle(args.graph_path)
    if args.verbose > 0:
        print('reading graph DONE')
    
    stime = time.time()
    if args.method == DetectionMethods.PR_ON_POS:
        other_params = {}
        pred_comm = get_comunity_using_pos_pagerank(
            g, args.query_node, args.teleport_alpha,
            **other_params
        )
    elif args.method == DetectionMethods.JUMPING_RW:
        other_params = dict(
            beta=10,
            gamma=0,
            truncate_percentile=75
        )
        pred_comm = get_community_by_jumping_pagerank(
            g, args.query_node, args.teleport_alpha,
            **other_params,
            verbose=args.verbose,
            show_progress=args.show_progress
        )

    time_elapsed = time.time() - stime
    print("community detection done")

    ans = OrderedDict()
    ans['graph_path'] = args.graph_path

    ans['method'] = args.method
    ans['query_node'] = args.query_node
    ans['teleport_alpha'] = args.teleport_alpha
    ans['other_params'] = other_params

    ans['community'] = pred_comm
    ans['conductance'] = signed_conductance(g, pred_comm)
    ans['time_elapsed'] = time_elapsed
    
    print(ans)
