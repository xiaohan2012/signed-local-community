import argparse
import networkx as nx
import time

from tqdm import tqdm

from sql import init_db, record_exists, insert_record, TableCreation
from core import query_graph_using_sparse_linear_solver, sweep_on_x_fast


KS = (200, 400, 800)


def query_given_two_seeds(
        g,
        seed1,
        seed2,
        kappa=0.9,
        ks=KS,
        verbose=0,
        **kwargs
):
    seeds = [[seed1], [seed2]]
    x, obj_val, runtime_info = query_graph_using_sparse_linear_solver(
        g, seeds, kappa=kappa, verbose=verbose,
        **kwargs
    )
    rows = []
    for k in ks:
        C1, C2, C, best_t, best_beta, ts, beta_array = sweep_on_x_fast(g, x, top_k=k)
        rows.append(
            dict(
                seed1=seed1,
                seed2=seed2,
                k=k,
                kappa=kappa,
                C1=C1,
                C2=C2,
                best_t=best_t,
                best_beta=best_beta,
                ts=ts,
                beta_array=beta_array,
                runtime_info=runtime_info
            )
        )
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Query in batch. 
    Assume one seed node in the each group
    """)
    parser.add_argument('-g', '--graph_path', required=True,
                        help='path of graph')
    parser.add_argument('--seed1_list', type=int, nargs='+',
                        required=True, help='list of seeds to run in the first group')
    parser.add_argument('--seed2_list', type=int, nargs='+',
                        required=True, help='list of seeds to run in the second group')
    parser.add_argument('-k', '--kappa', type=float,
                        default=0.9,
                        help='correlation coefficient')
    parser.add_argument('-v', '--verbose', type=int,
                        default=0,
                        help='verbose level (>= 0)')
    parser.add_argument('-p', '--show_progress',
                        action='store_true',
                        help='show progress or not')
    parser.add_argument('-d', '--save_to_db',
                        action='store_true',
                        help='save to db or not')

    args = parser.parse_args()

    g = nx.read_gpickle(args.graph_path)

    assert len(args.seed1_list) == len(args.seed2_list), 'should be equal length'
    
    for s1, s2 in tqdm(zip(args.seed1_list, args.seed2_list), total=len(args.seed1_list)):
        s1, s2 = sorted([s1, s2])  # s1 < s2
        stime = time.time()
        rows = query_given_two_seeds(
            g, seed1=s1, seed2=s2,
            kappa=args.kappa,
            verbose=args.verbose,
            ub=g.graph['lambda1'],
            return_details=True,
            max_iter=40
        )
        time_elapsed = time.time() - stime

        if args.save_to_db:
            conn, cursor = init_db(create_table=True)
        
            for row in rows:
                row['graph_path'] = args.graph_path
                row['time_elapsed'] = time_elapsed

                filter_value = dict(
                    graph_path=args.graph_path,
                    kappa=row['kappa'],
                    seed1=row['seed1'],
                    seed2=row['seed2'],
                    k=row['k']
                )
                if not record_exists(cursor, TableCreation.seed_pair_table, filter_value):
                    insert_record(
                        cursor, TableCreation.seed_pair_table, row
                    )
            conn.commit()
            print('inserted to db')
            conn.close()
        else:
            for row in rows:            
                print(row)


