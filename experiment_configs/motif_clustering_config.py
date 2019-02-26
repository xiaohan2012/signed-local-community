"""
experiment on motif clustering
"""

import sys
import os
import numpy as np
from .base import BaseConfig

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

from helpers import make_range, round_up, motif_primary_key_value
from sql import record_exists, TableCreation

np.random.seed(12345)


class ExperimentConfig(BaseConfig):
    def __init__(
            self,
            # input graph
            graph_path,

            # method related
            motifs=[],
            query_node=0,
            teleport_alpha=0.5,
            **kwargs
    ):
        self.graph_path                =    graph_path
        self.motifs                    =    motifs
        self.query_node                =    int(query_node)
        self.teleport_alpha            =    round_up(teleport_alpha, 1)

        
        self.cmd_arg_names = (
           'graph_path',
           'motifs',
           'query_node',
           'teleport_alpha'
        )

        super().__init__(**kwargs)

    def is_computed(self, cursor):
        record = {
            'id': motif_primary_key_value(self.graph_path, self.motifs, self.teleport_alpha, self.query_node)
        }
        return record_exists(cursor, TableCreation.query_result_table, record)
        
    def print_commands(self, fileobj=sys.stdout, prefix=''):
        params = [(a, getattr(self, a)) for a in self.cmd_arg_names]

        str_list = []
        for name, value in params:
            if isinstance(value, tuple):
                value = ' '.join(value)
            str_list.append('--{} {}'.format(name, value))

        arg_str = prefix + " "
        arg_str += ' '.join(str_list)
        arg_str += (' ' + self.suffix)
        fileobj.write(arg_str + '\n')
