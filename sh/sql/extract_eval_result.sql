WITH
    subset AS (
        SELECT * FROM signed_local_comm_dbg.eval_result WHERE graph_path = 'graphs/bitcoin.pkl'
    ),
    n_nodes AS (    
        SELECT
            id, method,	value
	FROM
	    subset
	WHERE
	    key = 'n'
    ),
    n_edges AS (    
        SELECT
            id, method,	value
	FROM
	    subset
	WHERE
	    key = 'm'
    ),
    inter_neg AS (
        SELECT
            id, method,	value
	FROM
	    subset
	WHERE
	    key = 'inter_neg_edges'    
    ),
    intra_pos AS (
        SELECT
            id, method,	value
	FROM
	    subset
	WHERE
	    key = 'intra_pos_edges'    
    ),    
    f1 AS (    
        SELECT
            id, method,	query_node, value
	FROM
	    subset
	WHERE
	    key = 'f1_pos_neg'
    ),
    avg_cc AS (    
        SELECT
            id, method,	value
	FROM
	    subset
	WHERE
	    key = 'avg_cc'
    ),
    diameter AS (    
        SELECT
            id, method,	value
	FROM
	    subset
	WHERE
	    key = 'diameter'
    )
SELECT
    f1.id, f1.method, f1.query_node, 
    n_nodes.value as n_nodes, n_edges.value as n_edges,
    inter_neg.value as inter_neg, intra_pos.value as intra_pos, f1.value AS f1_pos_neg,
    avg_cc.value AS avg_cc, diameter.value AS diameter    
FROM
    f1
JOIN
   avg_cc
ON
   f1.id = avg_cc.id AND f1.method = avg_cc.method
JOIN
   diameter
ON
   f1.id = diameter.id AND f1.method = diameter.method
JOIN
   n_nodes
ON
   f1.id = n_nodes.id AND f1.method = n_nodes.method
JOIN
   n_edges
ON
   f1.id = n_edges.id AND f1.method = n_edges.method
JOIN
   inter_neg
ON
   f1.id = inter_neg.id AND f1.method = inter_neg.method
JOIN
   intra_pos
ON
   f1.id = intra_pos.id AND f1.method = intra_pos.method      
