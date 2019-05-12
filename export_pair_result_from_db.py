import pandas as pd
from sql import init_db
import pickle as pkl
from tqdm import tqdm

conn, cursor = init_db()

graph = 'bitcoin'
cursor.execute("""
    SELECT 
        seed1, seed2, k, C1, C2
    FROM local_polarization.query_result_seed_pair
    WHERE graph_path LIKE '%%{}%%'
""".format(graph)
)

cols = (
    'seed1', 'seed2', 'k', 'C1', 'C2'
)
rows = []
for r in tqdm(cursor.fetchall()):
    r = list(r)
    r[3] = pkl.loads(r[3])
    r[4] = pkl.loads(r[4])
    rows.append(r)
df = pd.DataFrame(rows, columns=cols)
df.to_pickle('outputs/{}_seed_pair.pkl'.format(graph))
