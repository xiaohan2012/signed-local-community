import psycopg2
import json
import numpy as np
import pickle as pkl

from const import DEBUG, DB_CONNECTION_STRING


class TableCreation:
    """
    configuration for the database-related stuff
    """
    
    if DEBUG:
        schema = 'signed_local_comm_dbg'
    else:
        schema = 'signed_local_comm'        

    query_result_table = 'query_result'
    query_result_table_creation = """
    CREATE TABLE IF NOT EXISTS {schema}.{table_name}
    (
        id                     TEXT,

        graph_path             TEXT,
        method                 TEXT,
        query_node             INTEGER,
        teleport_alpha         NUMERIC,
        other_params           JSONB,

        community              BYTEA
,
        time_elapsed           REAL
    )
    """.format(
        table_name=query_result_table,
        schema=schema
    )
    
    eval_result_table = 'eval_result'
    eval_result_table_creation = """
    CREATE TABLE IF NOT EXISTS {schema}.{table_name}
    (
        id                     TEXT,

        graph_path             TEXT,
        method                 TEXT,
        query_node             INTEGER,
        teleport_alpha         NUMERIC,
        other_params           JSONB,

        key                    TEXT,
        value                  REAL
    )
    """.format(
        table_name=eval_result_table,
        schema=schema
    )


def init_db(debug=False):
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
    for sql in sqls_to_execute:
        cursor.execute(sql)
    if debug:
        conn.set_trace_callback(print)
        pass
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
    
    
def insert_record(cursor, schema, table, record):
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
        schema=schema,
        table_name=table,
        fields=', '.join(record.keys()),
        placeholders=', '.join(['%s'] * len(record))
    ),
        tuple(record.values())
    )
    
