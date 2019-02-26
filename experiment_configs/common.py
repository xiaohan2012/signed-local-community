
import numpy as np
from itertools import product
from tqdm import tqdm


def make_iter_configs(config_dimensions, config_class):
    def aux(show_progress=False):
        if show_progress:
            total = np.prod([len(c) for c in config_dimensions])
            iters = tqdm(product(*config_dimensions), total=total)
        else:
            iters = product(*config_dimensions)
        for dict_list in iters:
            params = {}
            for d in dict_list:
                params.update(d)
        
            yield config_class(**params)

    return aux
