"""
Effect of ratio of internal negative edges
"""

import sys
import os
import numpy as np
from itertools import product
from collections import namedtuple
from tqdm import tqdm
from .base import BaseConfig

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

from helpers import make_range, round_up
from experiment_on_community_graph import get_graph_path
from sql import record_exists, TableCreation

np.random.seed(12345)


class CommunityGraphBaseConfig(BaseConfig):
    def __init__(
            self,
            # input graph
            graph_id,
            community_size=16,
            num_communities=4,
            internal_density=0.8,
            internal_negative_ratio=0.2,
            external_edge_proba=0.1,
            external_neg_ratio=0.9,

            # method related
            method=None,
            query_node=0,
            teleport_alpha=0.5,
            **kwargs
    ):
        self.graph_id                  =    int(graph_id)
        self.community_size            =    int(community_size)
        self.num_communities           =    int(num_communities)
        self.internal_density          =    round_up(internal_density, 1)
        self.internal_negative_ratio   =    round_up(internal_negative_ratio, 1)
        self.external_edge_proba       =    round_up(external_edge_proba, 1)
        self.external_neg_ratio        =    round_up(external_neg_ratio, 1)

        self.method                    =    method
        self.query_node                =    int(query_node)
        self.teleport_alpha            =    round_up(teleport_alpha, 1)

        self.cmd_arg_names = (
           'graph_id',
           'community_size',
           'num_communities',
           'internal_density',
           'internal_negative_ratio',
           'external_edge_proba',
           'external_neg_ratio',
           'method',
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
            'method',
            'query_node',
            'teleport_alpha'
        )

        super().__init__(**kwargs)
        
    def is_computed(self, cursor):
        record = {a: getattr(self, a)
                  for a in self.filter_arg_names}
        return record_exists(cursor, TableCreation.comm_graph_exp_table, record)
        
    def print_commands(self, fileobj=sys.stdout, prefix=''):
        params = [(a, getattr(self, a)) for a in self.cmd_arg_names]

        str_list = []
        for name, value in params:
            str_list.append('--{} {}'.format(name, value))

        arg_str = prefix + " "
        arg_str += ' '.join(str_list)
        fileobj.write(arg_str + '\n')


