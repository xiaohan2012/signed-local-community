WITH
    subset AS (
        SELECT * FROM signed_local_comm_dbg.eval_result WHERE graph_path = 'graphs/congress.pkl'
    ),
    f1 AS (    
        SELECT
            id, method,	value
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
    f1.id, f1.method, f1.value AS f1_pos_neg, avg_cc.value AS avg_cc, diameter.value AS diameter
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
