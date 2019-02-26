import sys
import os

from .motif_clustering_config import ExperimentConfig
from .common import make_iter_configs

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from motif_adjacency import M1, M2, M3
from const import ALL_GRAPHS_AND_NUM_NODES

motif_combinations = [
    (M1, ),
    (M2, ),
    (M3, ),
    (M1, M2),
    (M1, M2, M3)
]

config_dimensions = [
    [dict(graph_path=path, query_node=q)
     for path, n in ALL_GRAPHS_AND_NUM_NODES for q in range(n)],
    [dict(motifs=m) for m in motif_combinations],
    [dict(hours_per_job=1, suffix='--save_db')]
]

iter_configs = make_iter_configs(
    config_dimensions, ExperimentConfig
)
