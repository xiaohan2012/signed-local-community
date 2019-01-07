WITH
    f1_per_experiment AS (
        SELECT
            graph_params->>'internal_negative_ratio' AS internal_neg_ratio, method, query_node, max(f1) AS f1
        FROM
            signed_local_comm_dbg.community_graph_experiment
        GROUP BY
            graph_path, graph_params->>'internal_negative_ratio', method, query_node
    )
SELECT
    internal_neg_ratio, method, avg(f1) AS f1_avg, stddev(f1) AS f1_stddev, count(1) AS f1_support
FROM
    f1_per_experiment
GROUP BY
    internal_neg_ratio, method
ORDER BY
    internal_neg_ratio, method
