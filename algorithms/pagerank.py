import numpy as np
import networkx as nx


def pr_score(g, q, alpha):
    personalization = {n: 0.0 for n in g.nodes()}
    personalization[q] = 1.0
    pr = nx.pagerank(g, alpha=1-alpha, personalization=personalization)
    scores = np.zeros(g.number_of_nodes())
    for v, s in pr.items():
        scores[v] = s
    return scores
