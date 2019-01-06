
import numpy as np
from itertools import product
from tqdm import tqdm


def make_iter_configs(config_dimensions, config_class):
    def aux():
        total = np.prod([len(c) for c in config_dimensions])
        for dict_list in tqdm(product(*config_dimensions), total=total):
            params = {}
            for d in dict_list:
                params.update(d)
        
            yield config_class(**params)

    return aux
