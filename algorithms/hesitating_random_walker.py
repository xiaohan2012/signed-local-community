import networkx as nx
from .pagerank import pr_score
from .sweep_cut import get_community


def modify_graph(g):
    mod_g = nx.MultiDiGraph()

    mod_g.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if g[u][v]['sign'] > 0:
            if g.is_directed():
                mod_g.add_edge(u, v)
            else:
                mod_g.add_edge(u, v)
                mod_g.add_edge(v, u)
        else:
            if g.is_directed():
                mod_g.add_edges(u, u)
            else:
                mod_g.add_edge(u, u)
                mod_g.add_edge(v, v)
    return mod_g


def run(g, query, alpha):
    mod_g = modify_graph(g)
    scores = pr_score(mod_g, query, alpha)
    return get_community(g, scores)
