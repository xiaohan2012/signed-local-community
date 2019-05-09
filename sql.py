import psycopg2
import json
import numpy as np
import pickle as pkl

from const import DB_CONNECTION_STRING


class TableCreation:
    """
    configuration for the database-related stuff
    """
    
    schema = 'local_polarization'

    query_result_table = 'query_result'
    query_result_table_creation = """
    CREATE TABLE IF NOT EXISTS {schema}.{table_name}
    (
        graph_path             TEXT,
        query                  INTEGER,
        kappa                  NUMERIC,
        k                      INTEGER,

        C1                     BYTEA,
        C2                     BYTEA,
        best_beta              REAL,
        best_t                 REAL,
        beta_array             BYTEA,
        ts                     BYTEA,

        time_elapsed           REAL,
        runtime_info           BYTEA
    );
    CREATE INDEX IF NOT EXISTS {schema}_{table_name}_idx ON {schema}.{table_name} (graph_path, query, kappa, k);
    """.format(
        table_name=query_result_table,
        schema=schema
    )
    
    eval_result_table = 'eval_result'
    eval_result_table_creation = """
    CREATE TABLE IF NOT EXISTS {schema}.{table_name}
    (
        graph_path             TEXT,
        query                  INTEGER,
        kappa                  NUMERIC,
        k                      INTEGER,

        key                    TEXT,
        value                  REAL
    );
    CREATE INDEX IF NOT EXISTS {schema}_{table_name}_idx ON {schema}.{table_name} (graph_path, query, kappa, k, key);
    """.format(
        table_name=eval_result_table,
        schema=schema
    )


def init_db(debug=False, create_table=False):
    """
    create connection, make a cursor and create the tables if needed
    """
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()

    cursor.execute(
        """CREATE SCHEMA IF NOT EXISTS {}""".format(TableCreation.schema)
    )
    sqls_to_execute = (
        TableCreation.query_result_table_creation,
        TableCreation.eval_result_table_creation
    )
    if create_table:
        for sql in sqls_to_execute:
            cursor.execute(sql)
        conn.commit()
        print('execution SQL done')
    if debug:
        conn.set_trace_callback(print)

    return conn, cursor


def record_exists(cursor, table, record):
    """record: dict"""
    cursor.execute(
        """
    SELECT 1 FROM
        {schema}.{table_name}
    WHERE
        {filter_template}
    """.format(
        schema=TableCreation.schema,
        table_name=table,
        filter_template=' AND '.join(
            map(lambda s: "{}=%s".format(s), record.keys())
        )
    ),
        tuple(record.values())
    )    
    return cursor.fetchone() is not None
    
    
def insert_record(cursor, table, record):
    # convert complex type to pickle
    for k, v in record.items():
        if isinstance(v, dict):
            record[k] = json.dumps(v)

        if isinstance(v, (list, tuple, set, np.ndarray)):
            record[k] = pkl.dumps(v)

    cursor.execute(
        """
    INSERT INTO
        {schema}.{table_name} ({fields})
    VALUES
        ({placeholders})
    """.format(
        schema=TableCreation.schema,
        table_name=table,
        fields=', '.join(record.keys()),
        placeholders=', '.join(['%s'] * len(record))
    ),
        tuple(record.values())
    )
    
