import pytest
import networkx as nx
from eval_helpers import (
    edge_agreement_ratio,
    avg_cc,
    cohesion, avg_cohesion,
    opposition, avg_opposition
)
from tqdm import tqdm


@pytest.fixture
def g():
    graph_name = 'toy'
    path = 'graphs/{}.pkl'.format(graph_name)
    return nx.read_gpickle(path)


def test_edge_agreement_ratio(g):
    bags_of_groups = [
        [[0, 1], [2]],
        [[0, 1], [5]],
        [[1, 5], [0]]
        
    ]
    expected_agreement_ratios = [1.0, 2/3, 0]

    for groups, expected_agreement_ratio in tqdm(
            zip(bags_of_groups, expected_agreement_ratios)):
        assert edge_agreement_ratio(g, groups) == expected_agreement_ratio


def test_avg_cc(g):
    bags_of_groups = [
        [[0, 1, 3]],
        [[0, 1, 3], [2]],
        [[0, 1, 3], [2], [5]],
        [[0, 1, 5]],
        [[0, 1, 5], [2]]
    ]
    expected_avg_ccs = [1, 3/4, 3/5, 0, 0]

    for groups, expected_avg_cc in tqdm(zip(bags_of_groups, expected_avg_ccs)):
        assert avg_cc(g, groups) == expected_avg_cc


def test_cohesion(g):
    test_cases = [([0, 1, 3], 1.0),
                  ([0, 1, 2], 1/3),
                  ([0, 1, 5], 2/3),
                  ([0, 1, 3, 4], 4 / 6)
    ]
    for grp, exp in test_cases:
        assert cohesion(g, grp) == exp


def test_avg_cohesion(g):
    test_cases = [
        ([(0, 1, 3), (0, 1, 2)], 2/3)
    ]

    for groups, exp in test_cases:
        assert avg_cohesion(g, groups) == exp


def test_opposition(g):
    test_cases = [
        (([0], [2]), 1.0),
        (([0, 1], [2]), 1.0),
        (([0], [5]), 0),
        (([0, 1], [5]), 0.5)
    ]
    for (grp1, grp2), exp in test_cases:
        act = opposition(g, grp1, grp2)
        assert act == exp, "{} != {}".format(act, exp)


def test_avg_opposition(g):
    test_cases = [
        ([(0, 1), (2, )], 1.0),
        ([(0, 1), (2, ), (3, )], 1/3)
    ]

    for groups, exp in test_cases:
        act = avg_opposition(g, groups)
        assert act == exp, "{} != {}".format(act, exp)
