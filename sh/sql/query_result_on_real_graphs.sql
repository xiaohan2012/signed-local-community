WITH
    result_for_graph AS (
        SELECT
            *
        FROM
    	signed_local_comm_dbg.query_result
        WHERE
            graph_path LIKE '%slashdot1%'
    ),
    pg_res AS (
        SELECT
            query_node, method, size, conductance, purity
        FROM
            result_for_graph
        WHERE
            method = 'pagerank_on_pos_graph'
    ),
    jp_res AS (
        SELECT
            query_node, method, size, conductance, purity
        FROM
            result_for_graph
        WHERE
            method = 'jumping_pagerank'
    ),
    pg_and_jp AS (
	SELECT
	    pg_res.query_node,
	    pg_res.size AS pg_size,
	    jp_res.size AS jp_size,     
	    pg_res.conductance AS pg_conductance,
	    jp_res.conductance AS jp_conductance,
	    pg_res.purity AS pg_purity,
	    jp_res.purity AS jp_purity
	FROM
	    pg_res
	INNER JOIN
	    jp_res
	ON
	    pg_res.query_node = jp_res.query_node
	ORDER BY
	    pg_res.size ASC        
    ),
    size_info AS (
        SELECT
	   AVG(pg_size) AS pg_avg_size,
	   AVG(jp_size) AS jp_avg_size
	FROM
	   pg_and_jp
    ),
    num_wins_info AS (
        SELECT
           COUNT(pg_conductance > jp_conductance OR NULL) AS num_jp_wins_cond,
           COUNT(pg_conductance <= jp_conductance OR NULL) AS num_pg_wins_cond,
           COUNT(pg_purity <= jp_purity OR NULL) AS num_jp_wins_purity,
           COUNT(pg_purity > jp_purity OR NULL) AS num_pg_wins_purity
        FROM    
           pg_and_jp
    )
SELECT
    CAST(num_jp_wins_cond AS REAL) / (num_pg_wins_cond + num_jp_wins_cond) AS ratio_jp_wins_cond,
    CAST(num_jp_wins_purity AS REAL) / (num_pg_wins_purity + num_jp_wins_purity) AS ratio_jp_wins_purity,
    (num_pg_wins_cond + num_jp_wins_cond) AS support,
    jp_avg_size,
    pg_avg_size
FROM
    num_wins_info
CROSS JOIN
    size_info
