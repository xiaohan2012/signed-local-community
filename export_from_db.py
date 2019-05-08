import pandas as pd
from sql import init_db
import pickle as pkl
from tqdm import tqdm

conn, cursor = init_db()

graph = 'slashdot'
cursor.execute("""
    SELECT * FROM local_polarization.query_result
    WHERE graph_path LIKE '%%{}%%'
""".format(graph)
)

cols = (
    'graph_path', 'query', 'kappa', 'k', 'C1', 'C2',
    'best_beta', 'best_t', 'beta_array', 'ts', 'time_elapsed'
)
rows = []
for r in tqdm(cursor.fetchall()):
    r = list(r)
    r[2] = float(r[2])
    r[4] = pkl.loads(r[4])
    r[5] = pkl.loads(r[5])
    r[8] = pkl.loads(r[9])
    r[9] = pkl.loads(r[9])
    rows.append(r)
df = pd.DataFrame(rows, columns=cols)
sub_df = df[['query', 'C1', 'C2', 'k', 'best_beta']]
sub_df.to_pickle('outputs/{}.pkl'.format(graph))
