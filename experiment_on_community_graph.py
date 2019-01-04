import argparse
from parser_helper import (
    add_community_graph_args,
    add_pagerank_args,
    add_detection_methods_args
)
from const import DetectionMethod
from algorithms import (
    get_comunity_using_pos_pagerank,
    sweep_on_true_community
)


def load_community_graph_data(
        graph_id,
        k,
        n,
        internal_density,
        internal_negative_ratio,
        external_edge_proba,
        external_neg_ratio
):
    pass

def main():
    parser = argparse.ArgumentParser()
    add_community_graph_args(parser)
    add_pagerank_args(parser)
    add_detection_methods_args(parser)
    
    args = parser.parse_args()

    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('-' * 25)

    if args.detection_method == DetectionMethod.SWEEP_ON_TRUE:
        pass
    elif args.detection_method == DetectionMethod.PR_ON_POS:
        get_comunity_using_pos_pagerank(g)
    
if __name__ == '__main__':
    main()

