import numpy as np
import networkx as nx

from collections import OrderedDict

from sklearn.metrics import precision_recall_fscore_support, precision_score

from helpers import n_neg_edges, n_pos_edges


def frac_intra_neg_edges(subg, A):
    nodes = list(subg.nodes())
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    n_neg_edges_total = np.absolute(neg_A[nodes, :].sum()) / 2
    n_neg_edges_inside = n_neg_edges(subg)
    return n_neg_edges_inside / n_neg_edges_total


def frac_inter_pos_edges(subg, A):
    nodes = list(subg.nodes())
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    n_pos_edges_total = pos_A[nodes, :].sum() / 2
    n_pos_edges_inside = n_pos_edges(subg)
    return 1 - n_pos_edges_inside / n_pos_edges_total


def edge_agreement_ratio(subg, A):
    """number of good edges / total number of edges"""
    def n_cut_and_n_inside(A, nodes):
        n_inside = A[nodes, :][:, nodes].sum() / 2
        
        mask = np.zeros(A.shape[0], dtype=bool)
        mask[nodes] = 1
        n_cut = A[:, np.logical_not(mask)].sum()
        return n_inside, n_cut
        
    nodes = list(subg.nodes())
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    pos_A = pos_A.astype('bool')
    pos_A.eliminate_zeros()
    n_pos_edges_inside, n_pos_edges_cut = n_cut_and_n_inside(pos_A, nodes)

    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    neg_A = neg_A.astype('bool')
    neg_A.eliminate_zeros()
    n_neg_edges_inside, n_neg_edges_cut = n_cut_and_n_inside(neg_A, nodes)

    # print('n_pos_edges_inside, n_pos_edges_cut', n_pos_edges_inside, n_pos_edges_cut)
    # print('n_neg_edges_inside, n_neg_edges_cut', n_neg_edges_inside, n_neg_edges_cut)
    total = n_pos_edges_inside + n_pos_edges_cut + n_neg_edges_inside + n_neg_edges_cut
    n_good_edges = n_pos_edges_inside + n_neg_edges_cut

    agreement_ratio = n_good_edges / total
    return agreement_ratio


def community_summary(subg, g, A=None):
    if A is None:
        A = nx.adj_matrix(g, weight='sign')
    # neg_frac = 1 - frac_intra_neg_edges(subg, A)
    # pos_frac = 1 - frac_inter_pos_edges(subg, A)
    res = OrderedDict()
    res['n'] = subg.number_of_nodes()
    res['m'] = subg.number_of_edges()
    # res['inter_neg_ratio'] = neg_frac
    # res['intra_pos_ratio'] = pos_frac
    res['edge_agreement_ratio'] = edge_agreement_ratio(subg, A)

    # res['avg_cc'] = np.mean(list(nx.clustering(subg).values()))

    # if g.number_of_nodes() <= 1000:
    #     # for small graphs, compute diameter exactly
    #     res['diameter'] = nx.diameter(subg)
    # else:
    #     res['diameter'] = approx_diameter(g)
    return res


def evaluate_level_1(n, C_pred, C_true):
    """
    for evaluating local polarization detection

    consider nodes in C_tree has true label 1 and nodes in C_pred are predicted to have label 1,
    compute precision, recall and f1
    """
    y_pred = np.zeros(n)
    y_pred[C_pred] = 1
    y_true = np.zeros(n)
    y_true[C_true] = 1
    return precision_recall_fscore_support(y_true, y_pred, average='binary')


def evaluate_level_2(n, c1, c2, C_true, groups):
    """
    for evaluating local polarization detection

    consider only nodes in (c1 \cup c2) \cap C_true
    if it's in c1, it has label 1
    if it's in c2, it has label -1
    
    for nodes in groups, if it's in commmunity 1, it has label 1 and -1 otherwise

    compute the precision, recall and f1 for each community

    Params
    --------

    n: number of nodes
    c1: list of nodes in community 1
    c2: list of nodes in community 1
    C_true: list of nodes in community 1 or 2
    groups: list of two lists, each list corresponds to one community

    """
    c1 = list(set(C_true).intersection(set(c1)))
    c2 = list(set(C_true).intersection(set(c2)))
    
    y_pred = np.zeros(n)
    y_pred[c1] = 1
    y_pred[c2] = -1
    nodes = y_pred.nonzero()[0]
    y_pred = y_pred[nodes]

    y_true = np.zeros(n)
    for i, grp in enumerate(groups):
        y_true[grp] = i*2 - 1
    y_true = y_true[nodes]

    ret1 = precision_recall_fscore_support(y_true, y_pred, average='micro')
    # make sure evaluation is label invariant
    y_pred = -y_pred  # invert the labels
    ret2 = precision_recall_fscore_support(y_true, y_pred, average='micro')

    # return the best
    if ret1[2] > ret2[2]:
        return ret1
    else:
        return ret2


def mean_avg_precision(g, C1, C2, target_comm, true_groupings, verbose=0):
    """
    compute mean average precision on C1 (as label -1) and C2 (as label 1),
    the rest of the nodes have label 0
    """
    n = g.number_of_nodes()
    
    assert len(C1) > 0
    assert len(C2) > 0

    pred_y = np.zeros(n)
    pred_y[C1] = -1
    pred_y[C2] = 1

    true_y = np.zeros(pred_y.shape)
    true_y[true_groupings[target_comm][0]] = -1
    true_y[true_groupings[target_comm][1]] = 1

    prec1 = precision_score(true_y, pred_y, average=None)
    # reverse the assignment
    prec2 = precision_score(true_y, -pred_y, average=None)

    prec00, prec01 = prec1[0], prec1[2]
    prec10, prec11 = prec2[0], prec2[2]

    # one assigment dominates
    assert (prec00 >= prec10 and prec01 >= prec11) or (prec00 < prec10 and prec01 < prec11)
    
    # take the best
    prec0, prec1 = np.maximum(prec00, prec10), np.maximum(prec01, prec11)
    if len(C1) + len(C2) >= n:
        if verbose > 0:
            print('prec1', prec1)
            print('prec2', prec2)
            print('MAP:', (prec0 + prec1) / 2)
    return (prec0 + prec1) / 2

