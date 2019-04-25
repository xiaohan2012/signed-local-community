"""
main function: `build_graph`
"""

import networkx as nx
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def load_synonyms(lemma):
    synonyms = set()
    for syn in wn.synsets(lemma):
        for l in syn.lemmas():
            synonyms.add(l.name())
    return synonyms


def load_antonyms(lemma):
    antonyms = set()
    for syn in wn.synsets(lemma):
        for l in syn.lemmas():
            for ant in l.antonyms():
                antonyms.add(ant.name())
    return antonyms


def build_graph():
    """build the graph"""
    all_words = []
    with open('graphs/wordnet/google-10000-english.txt', 'r') as f:
        for l in f:
            all_words.append(l.strip())

    g = nx.Graph()
    g.add_nodes_from(all_words)
    for u in tqdm(all_words):
        for v in load_synonyms(u):
            g.add_edge(u, v, sign=1)
        for v in load_antonyms(u):
            g.add_edge(u, v, sign=-1)
    g.remove_edges_from(g.selfloop_edges())
    nx.write_gpickle(g, 'graphs/wordnet.pkl')

    
