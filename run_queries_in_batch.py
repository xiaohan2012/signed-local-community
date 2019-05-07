import argparse
import networkx as nx
from tqdm import tqdm
from exp_helpers import query_given_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query in batch')
    parser.add_argument('-g', '--graph', required=True,
                        help='path of graph')
    parser.add_argument('-q', '--query_list', type=int, nargs='+',
                        required=True, help='list of queries to run')
    parser.add_argument('-k', '--kappa', type=float,
                        default=0.9,
                        help='correlation coefficient')
    parser.add_argument('-v', '--verbose', type=int,
                        default=0,
                        help='verbose level (>= 0)')
    parser.add_argument('-p', '--show_progress',
                        action='store_true',
                        help='show progress or not')

    arg = parser.parse_args()

    g = nx.read_gpickle(arg.graph)
        
    for q in tqdm(arg.query_list):
        rows = query_given_seed(g, q, arg.kappa, arg.verbose)
