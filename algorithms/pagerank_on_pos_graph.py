import networkx as nx
from .pagerank import pr_score
from .sweep_cut import get_community


def extract_pos_graph(g):
    pos_g = nx.Graph()
    for u, v in g.edges():
        if g[u][v]['sign'] == 1.0:
            pos_g.add_edge(u, v)
    return pos_g


def run(g, query, alpha):
    pos_g = extract_pos_graph(g)
    scores = pr_score(pos_g, query, alpha)
    return get_community(g, scores)
