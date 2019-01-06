import sys
import os
import numpy as np

from .community_graphs_config import CommunityGraphBaseConfig
from .common import make_iter_configs

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from helpers import make_range
from const import DetectionMethods

config_dimensions = [
    [dict(graph_id=v) for v in range(10)],
    [dict(internal_negative_ratio=v) for v in make_range(0, 0.5)],
    [dict(method=m) for m in [DetectionMethods.SWEEP_ON_TRUE, DetectionMethods.PR_ON_POS]],
    [dict(query_node=v) for v in np.random.permutation(4 * 16)[:8]],
    [dict(teleport_alpha=v) for v in make_range(0.1, 0.9)],
]

iter_configs = make_iter_configs(
    config_dimensions, CommunityGraphBaseConfig
)
