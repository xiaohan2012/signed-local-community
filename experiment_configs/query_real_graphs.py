import sys
import os
import numpy as np

from .query_real_graphs_config import QueryingRealGraphConfig
from .common import make_iter_configs

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from const import DetectionMethods

ALL_GRAPHS_AND_NUM_NODES = [
    # ('graphs/tribe.pkl', 16)
    ('graphs/slashdot1/graph.pkl', 77268)
]
DETECTION_METHODS = [
    DetectionMethods.PR_ON_POS,
    DetectionMethods.JUMPING_RW
]

config_dimensions = [
    [dict(graph_path=path, query_node=q)
     for path, n in ALL_GRAPHS_AND_NUM_NODES for q in range(n)],
    [dict(method=m) for m in DETECTION_METHODS],
    [dict(hours_per_job=5)]
]

iter_configs = make_iter_configs(
    config_dimensions, QueryingRealGraphConfig
)
