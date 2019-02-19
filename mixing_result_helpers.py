import numpy as np
import networkx as nx


from numpy.linalg import eigh as numpy_eigh
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt


def flatten(stuff):
    return np.asarray(stuff).flatten()


def normalized_laplacian(g):
    assert not g.is_directed()
    A = nx.adjacency_matrix(g).toarray()
    deg = A.sum(axis=0)
    D_neg_half = np.diag(flatten(1 / np.sqrt(deg)))
    L_norm = np.eye(A.shape[0]) - D_neg_half.dot(A).dot(D_neg_half)
    return L_norm


def conductance(g, S, verbose=False):
    numer = 0
    denum = 0
    S = set(S)
    for u in S:
        for v in g.neighbors(u):
            if v not in S:
                numer += 1
            denum += 1
    denum = min(denum, 2 * g.number_of_edges() - denum)
    if verbose >= 1:
        print('{} / {}'.format(numer, denum))
    return numer / denum


def load_example_graphs():
    complete_graph = nx.complete_graph(16)
    complete_graph.graph['name'] = 'complete(16)'
    complete_graph.graph['phi'] = min(conductance(complete_graph, list(range(i))) for i in range(1, 16))

    lattice = nx.convert_node_labels_to_integers(
            nx.lattice.grid_2d_graph(4, 4)
    )
    lattice.graph['name'] = 'lattice(4,4)'
    lattice.graph['phi'] = min(conductance(lattice, list(range(i))) for i in range(1, 16))

    barbell_graph = nx.barbell_graph(8, 8)
    barbell_graph.graph['name'] = 'barbell(8,8)'
    barbell_graph.graph['phi'] = conductance(barbell_graph, list(range(8)))

    line_graph = nx.path_graph(16)
    line_graph.graph['name'] = 'line(16)'
    line_graph.graph['phi'] = min(conductance(line_graph, list(range(i))) for i in range(1, 16))

    for g in [barbell_graph, line_graph, lattice, complete_graph]:
        l2 = numpy_eigh(normalized_laplacian(g))[0][1]
        g.graph['lambda2'] = l2

    return barbell_graph, line_graph, lattice, complete_graph


def init_p0(g, seed=0):
    n = g.number_of_nodes()
    p0 = np.zeros((1, n))
    p0[0, seed] = 1
    return p0.transpose()


def get_relevant_matrices(g):
    """returns:

    - lazy random walk matrix
    - the socket matrix
    """
    n, m = g.number_of_nodes(), g.number_of_edges()
    A = nx.adjacency_matrix(g).toarray()
    deg = A.sum(axis=0)
    Di = np.diag(1 / deg)
    Wl = (np.eye(n) + A.dot(Di)) / 2  # make it lazy

    ai, aj = np.nonzero(A)  # zip(ai, aj) = edges
    sock = csr_matrix((1 / deg[ai], (np.arange(0, 2 * m), ai))).todense()  # non-lazy version
    sock = (np.vstack((sock, sock)) / 2)  # lazy version, the 2nd block are the self-loops

    return Wl, sock


def plot_Ct_list(Wl, sock, p0, alpha, ax, log=False, k=100, step=10):
    p_cur = p0
        
    lines = []
    for i in range(0, k, step):
        C = np.cumsum(np.sort(flatten(sock @ p_cur))[::-1])
        if log:
            C = np.log2(C)
            
        ls = ax.plot(np.arange(C.shape[0]+1), [0] + C.tolist(), color='orange', alpha=0.6)
        lines.append(ls[0])
        ax.hold(True)
        
        p_cur = alpha * p0 + (1 - alpha) * Wl @ p_cur  # do the PPR

    return lines


def Ut(x, t, m, phi, alpha, **kwargs):
    return (x / 4 / m + alpha * t
            + np.minimum(np.sqrt(x), np.sqrt(4 * m - x)) * ((1 - np.power(phi, 2) / 8) ** t))


def plot_Ut_list(m, phi_G, alpha, ax, Ut_func=Ut, k=100, step=10, log=False, **Ut_kwargs):
    x = np.arange(0, 4 * m+1)

    lines = []
    
    for t in range(0, k, step):
        Ui = Ut_func(x, t, m, phi_G, alpha, **Ut_kwargs)
        if log:
            Ui = np.log2(Ui)
        
        ls = ax.plot(np.arange(len(Ui)), Ui, color='blue', alpha=0.6)
        lines.append(ls[0])
        ax.hold(True)

    return lines


def plot_curves(g, seed, alpha, k=50, step=10, use_log=False, ax=None):
    n, m = g.number_of_nodes(), g.number_of_edges()

    print('n/m={}/{}'.format(n, m))

    Wl, sock = get_relevant_matrices(g)

    p0 = init_p0(g, seed)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    Ut_lines = plot_Ut_list(m, g.graph['phi'], alpha, ax, k=k, step=step, log=use_log)
    Ct_lines = plot_Ct_list(Wl, sock, p0, alpha, ax, k=k, step=step, log=use_log)
    
    ax.set_xlabel('x')
    if use_log:
        ax.set_ylabel('log(value)')
    else:
        ax.set_ylabel('value')
    ax.set_title('{}, $\phi_G={:.3f}$'.format(g.graph['name'], g.graph['phi']))
    ax.legend([Ct_lines[0], Ut_lines[0]], ['$C_t$', '$U_t$'], loc='best')


def plot_Ct_list_in_order(Wl, sock, p0, alpha, ax, log=False, k=100, step=10, cm_name='Oranges'):
    """
    add different colors to the lines
    """
    p_cur = p0

    cm = plt.get_cmap(cm_name)
    colors = cm(list((np.arange(k) + k / 3) / (k + k / 3)))  # cm accepts value from 0 to 1
    
    lines = []
    for i in range(0, k, step):
        C = np.cumsum(np.sort(flatten(sock @ p_cur))[::-1])
        if log:
            C = np.log2(C)
            
        ls = ax.plot(np.arange(C.shape[0]+1), [0] + C.tolist(), color=colors[i, :], alpha=0.6)
        lines.append(ls[0])
        ax.hold(True)
        
        p_cur = alpha * p0 + (1 - alpha) * Wl @ p_cur  # do the PPR

    return lines


def plot_Ut_list_in_order(
    m, phi_G, alpha, ax, Ut_func=Ut, k=100, step=10, log=False, cm_name='Blues', **Ut_kwargs
):
    """
    add different colors to the lines
    """        
    x = np.arange(0, 4 * m+1)

    cm = plt.get_cmap(cm_name)
    colors = cm(list((np.arange(k) + k / 2) / (k + k / 2)))  # cm accepts value from 0 to 1
    lines = []
    
    for t in range(0, k, step):
        Ui = Ut_func(x, t, m, phi_G, alpha, **Ut_kwargs)
        if log:
            Ui = np.log2(Ui)
        
        ls = ax.plot(np.arange(len(Ui)), Ui, color=colors[t, :], alpha=0.6)
        lines.append(ls[0])
        ax.hold(True)

    return lines
