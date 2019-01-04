import sys
import os
import numpy as np
from itertools import product
from collections import namedtuple
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

from helpers import make_range
from const import DetectionMethods
from experiment_on_community_graph import get_graph_path
from sql import record_exists, TableCreation

np.random.seed(12345)


class Config:
    def __init__(
            self,
            # input graph
                 graph_id,
            community_size,
            num_communities,
            internal_density,
            internal_negative_ratio,
            external_edge_proba,
            external_neg_ratio,

            # method related
            method,
            query_node,
            teleport_alpha
    ):
        self.graph_id                  =    graph_id
        self.community_size            =    community_size     
        self.num_communities           =    num_communities        
        self.internal_density          =    internal_density       
        self.internal_negative_ratio   =    internal_negative_ratio
        self.external_edge_proba       =    external_edge_proba    
        self.external_neg_ratio        =    external_neg_ratio

        self.detection_method          =    method        
        self.query_node                =    query_node    
        self.teleport_alpha            =    teleport_alpha

        self.cmd_arg_names = (
           'graph_id',
           'community_size',
           'num_communities',
           'internal_density',
           'internal_negative_ratio',
           'external_edge_proba',
           'external_neg_ratio',
           'detection_method',
           'query_node',
           'teleport_alpha'
        )

        Args = namedtuple('Args', list(self.cmd_arg_names))

        filter_args = {a: getattr(self, a)
                       for a in self.cmd_arg_names}
        self.graph_path = get_graph_path(
            Args(**filter_args)
        )

        self.filter_arg_names = (
            'graph_path',
            'detection_method',
            'query_node',
            'teleport_alpha'
        )

    def is_computed(self, cursor):
        record = {a: getattr(self, a)
                  for a in self.filter_arg_names}
        return record_exists(cursor, TableCreation.comm_graph_exp_table, record)
        
    def print_cmds(self, fileobj, prefix):
        params = [(a, getattr(self, a)) for a in self.cmd_arg_names]

        str_list = []
        for name, value in params:
            str_list.append('{} {}'.format(name, value))

        arg_str = prefix + " "
        arg_str += ' '.join(str_list)
        fileobj.write(arg_str + '\n')


config_dimensions = [
    [dict(community_size=16, num_communities=4)],
    [dict(graph_id=v) for v in range(10)],
    [dict(internal_density=v) for v in make_range(0.5, 1.0)],
    [dict(internal_negative_ratio=v) for v in make_range(0, 0.5)],
    [dict(external_edge_proba=v) for v in  make_range(0.0, 0.5)],
    [dict(external_neg_ratio=v) for v in make_range(0.6, 1.0)],
    [dict(method=m) for m in [DetectionMethods.SWEEP_ON_TRUE, DetectionMethods.PR_ON_POS]],
    [dict(query_node=v) for v in np.random.permutation(4 * 16)[:8]],
    [dict(teleport_alpha=v) for v in make_range(0.1, 0.9)],
]


def iter_configs():
    total = np.prod([len(c) for c in config_dimensions])
    for dict_list in tqdm(product(*config_dimensions), total=total):
        params = {}
        for d in dict_list:
            params.update(d)
    
        yield Config(**params)
