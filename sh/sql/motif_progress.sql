select
    graph_path, method, count(1) as cnt
from
    signed_local_comm_dbg.query_result
group by
    graph_path, method
order by
    graph_path, method
