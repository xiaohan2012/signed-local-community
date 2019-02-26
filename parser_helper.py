from const import ALL_DETECTION_METHODS


def add_real_graph_args(parser):
    parser.add_argument('-g', '--graph_path',
                        required=True,
                        type=str,
                        help='path of graph')

    
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
                        default=0.5,
                        help='probability to teleport')

    parser.add_argument('--max_iter',
                        type=int,
                        default=10,
                        help='maximum number of iterations')


def add_detection_methods_args(parser):
    parser.add_argument('-m', '--method',
                        choices=ALL_DETECTION_METHODS,
                        required=True,
                        help='')

def add_motif_args(parser):
    parser.add_argument('-m', '--motifs',
                        required=True,
                        nargs='+',
                        type=str,
                        help='the list of motifs separated by space')
    
def add_misc_args(parser):
    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='verbose level')
    parser.add_argument('--show_progress',
                        action='store_true',
                        help='show progress bar or not')
    parser.add_argument('--save_db',
                        action='store_true',
                        help='store to db or not')
    parser.add_argument('-experiment_id',
                        type=str,
                        help='the unique experiment id')
