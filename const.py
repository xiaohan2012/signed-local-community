import inspect


DEBUG = True

DATA_DIR = "/scratch/work/xiaoh1/data/signed-local-community/"
TMP_DIR = "/scratch/work/xiaoh1/data/signed-local-community/tmp"


class DetectionMethods:
    SWEEP_ON_TRUE = 'sweep_on_true'
    PR_ON_POS = 'pagerank_on_pos_graph'
    JUMPING_RW = 'jumping_pagerank'

attrs = inspect.getmembers(DetectionMethods, lambda a: not(inspect.isroutine(a)))
ALL_DETECTION_METHODS = tuple([a[1]
                               for a in attrs
                               if not(a[0].startswith('__') and a[0].endswith('__'))])

DB_CONNECTION_STRING = 'dbname=postgres user=xiaoh1 host=10.10.254.21'

ALL_GRAPHS_AND_NUM_NODES = [
    # ('graphs/tribe.pkl', 16),
    ('graphs/cloister.pkl', 18),
    # ('graphs/congress.pkl', 219),
    # ('graphs/bitcoin.pkl', 5881),
    # ('graphs/thesaurus.pkl', 15670),
    ('graphs/slashdot1.pkl', 77268)
]
