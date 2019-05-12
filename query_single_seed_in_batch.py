import argparse
import networkx as nx
import time

from tqdm import tqdm

from sql import init_db, record_exists, insert_record, TableCreation
from core import query_graph_using_sparse_linear_solver, sweep_on_x_fast


# KS = (100, 200, 400, 800, 1600, 3200)
KS = (100, 200, 400, )
# KS = (200, )


def query_given_seed(
        g,
        query,
        kappa=0.9,
        ks=KS,
        verbose=0,
        **kwargs
):
    x, obj_val, runtime_info = query_graph_using_sparse_linear_solver(
        g, [[query]], kappa=kappa, verbose=verbose,
        **kwargs
    )
    rows = []
    for k in ks:
        C1, C2, C, best_t, best_beta, ts, beta_array = sweep_on_x_fast(g, x, top_k=k)
        rows.append(
            dict(
                query=query,
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
    parser = argparse.ArgumentParser(description='Query in batch')
    parser.add_argument('-g', '--graph_path', required=True,
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
    parser.add_argument('-d', '--save_to_db',
                        action='store_true',
                        help='save to db or not')

    args = parser.parse_args()

    g = nx.read_gpickle(args.graph_path)

    for q in tqdm(args.query_list):
        stime = time.time()
        rows = query_given_seed(
            g, q, kappa=args.kappa, verbose=args.verbose,
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
                    query=row['query'],
                    k=row['k']
                )
                if not record_exists(cursor, TableCreation.single_seed_table, filter_value):
                    insert_record(
                        cursor, TableCreation.single_seed_table, row
                    )
            conn.commit()
            print('inserted to db')
            conn.close()
        else:
            for row in rows:            
                print(row)


