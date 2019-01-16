from const import ALL_DETECTION_METHODS


def add_community_graph_args(parser):
    parser.add_argument('-i', '--graph_id',
                        required=True,
                        type=int,
                        help='id of the graph')
    parser.add_argument('-k', '--num_communities',
                        required=True,
                        type=int,
                        help='number of communities')
    parser.add_argument('-s', '--community_size',
                        required=True,
                        type=int,
                        help='num. nodes in each community')
    parser.add_argument('-d', '--internal_density',
                        type=float,
                        required=True,
                        help='')
    parser.add_argument('-n', '--internal_negative_ratio',
                        type=float,
                        required=True,
                        help='')
    parser.add_argument('-p', '--external_edge_proba',
                        type=float,
                        required=True,
                        help='')
    parser.add_argument('-e', '--external_neg_ratio',
                        type=float,
                        required=True,
                        help='')

    parser.add_argument('-g', '--graph_dir',
                        type=str,
                        default='./graphs',
                        help='directory of graph data')


def add_pagerank_args(parser):
    parser.add_argument('-q', '--query_node',
                        type=int,
                        help='the query node')

    parser.add_argument('--teleport_alpha',
                        type=float,
                        help='probability to teleport')


def add_detection_methods_args(parser):
    parser.add_argument('-m', '--method',
                        choices=ALL_DETECTION_METHODS,
                        required=True,
                        help='')
