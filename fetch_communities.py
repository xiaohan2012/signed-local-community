import pandas as pd
import pickle as pkl
from sql import init_db, TableCreation

conn, cursor = init_db()

graph_name = 'thesaurus'

sql = """
    SELECT
      query_node, community
    FROM
      {schema}.{query_tbl}
    WHERE
      graph_path = 'graphs/{graph}.pkl'
      AND method = 'motif-m1m2'
""".format(
    schema=TableCreation.schema,
    query_tbl=TableCreation.query_result_table,
    graph=graph_name
)

print(sql)
cursor.execute(sql)

query2comm = {}
for r in cursor.fetchall():
    query2comm[r[0]] = pkl.loads(r[1])
pkl.dump(
    query2comm,
    open('query_results/{}.pkl'.format(graph_name), 'wb')
)
