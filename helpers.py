import numpy as np
import random
import string
import datetime

import networkx as nx


from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt


def walk(g, s0, beta, n_steps, verbose=0):
    """
    when the walker travels a "-" edge, it simply jumps back to s0

    with beta probability to jump back to s0
    """
    counter = np.zeros(g.number_of_nodes())
    current = s0
    for _ in range(n_steps):
        counter[current] += 1
        if verbose >= 2:
            print('current: ', current)
        r = random.random()
        if r <= beta:
            # teleport to s0
            next_nbr = s0
        else:
            nbrs = list(g.neighbors(current))
            next_nbr = random.choice(nbrs)
            sign = g[current][next_nbr]['sign']
            if sign == -1:
                next_nbr = s0
        if verbose >= 2:
            print('edge sign: ', sign)
        current = next_nbr

    return counter / n_steps


def signed_conductance(g, S, verbose=0):
    """
    fraction of edges from nodes S that are either + outside or - inside
    
    |cut+(S, V\S) + #deg-(S)| / sum_s deg(s)
    """
    numer = 0
    denum = 0
    S = set(S)
    for u in S:
        for v in g.neighbors(u):
            if v in S and g[u][v]['sign'] == -1:
                numer += 1
            elif v not in S and g[u][v]['sign'] == 1:
                numer += 1
            denum += 1
    if verbose >= 1:
        print('{} / {}'.format(numer, denum))
    return numer / denum


def signed_laplacian(g):
    deg = nx.adjacency_matrix(g).sum(axis=0)
    D = diags(deg.tolist()[0], 0)
    A = nx.adjacency_matrix(g, weight='sign')
    L = D - A
    return L


def signed_layout(g):
    L = signed_laplacian(g)
    w, pos_array = eigs(L.asfptype(), k=2, which='SM')
    pos_array = np.real(pos_array)
    return {i: pos_array[i, :] for i in range(g.number_of_nodes())}


def draw_nodes(g, pos):
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_labels(g, pos)


def draw_edges(g, pos):
    pos_edges = [(u, v) for u, v in g.edges() if g[u][v]['sign'] == 1.0]
    neg_edges = [(u, v) for u, v in g.edges() if g[u][v]['sign'] == -1.0]
    nx.draw_networkx_edges(g, pos, pos_edges, style='solid', edge_color='blue')
    nx.draw_networkx_edges(g, pos, neg_edges, style='dashed', edge_color='red')


def show_result(g, pos, query, scores):
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]

    print('nodes sorted by PR score', order)
    print('PR scores after sorting', sorted_scores)

    fig, ax = plt.subplots(1, 1)    
    nx.draw_networkx_nodes(
        g, pos, node_color=np.log2((scores + 1e-5) * 1e5), cmap='Blues')
    nx.draw_networkx_labels(g, pos)
    draw_edges(g, pos)
    ax.set_title('query node {}'.format(query))

    # sweeping plot
    sweep_positions = []
    sweep_scores = []
    for i in range(1, len(order)+1):
        if scores[order[i-1]] == 0:
            break
        sweep_positions.append(i)
        s = signed_conductance(g, order[:i])
        sweep_scores.append(s)


    fig, ax = plt.subplots(1, 1)
    ax.plot(sweep_positions, sweep_scores)
    ax.set_title('query node {}'.format(query))
    ax.set_xlabel('sweeping position')
    ax.set_ylabel('signed conductance')
    
    # get the best community
    best_pos = np.argmin(sweep_scores)
    comm = order[:best_pos+1]
    print('best position', best_pos)
    print('community', comm)
    
    fig, ax = plt.subplots(1, 1)
    color = np.zeros(g.number_of_nodes())
    color[comm] = 1
    nx.draw_networkx_nodes(g, pos, node_color=color, cmap='Blues')
    nx.draw_networkx_labels(g, pos)
    draw_edges(g, pos)
    ax.set_title('query node {}'.format(query))


def evaluate_performance(g, pred_comm, true_comm):
    prec = len(pred_comm.intersection(true_comm)) / len(true_comm)
    recall = len(pred_comm.intersection(true_comm)) / len(true_comm)
    f1 = 2 * prec * recall / (prec + recall)

    c = signed_conductance(g, pred_comm)

    return dict(prec=prec, recall=recall, f1=f1, conductance=c)


def get_now():
    return datetime.date.today().strftime("%Y-%m-%d %H:%M:%s")


def make_range(start, end, step=0.1):
    return np.arange(start, end + 0.1 * step, step)


def random_str(N=8):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return round(n * multiplier) / multiplier
