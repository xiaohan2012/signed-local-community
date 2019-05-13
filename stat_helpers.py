import numpy as np
from helpers import sbr


def avg_pos_deg(pos_A, nodes):
    """average positive degree"""
    return pos_A[nodes, :][:, nodes].sum() / nodes.shape[0]


def avg_neg_deg(neg_A, nodes):
    """average negative degree"""
    return neg_A[nodes, :][:, nodes].sum() / nodes.shape[0]


def cohe(pos_A, C1, C2):
    """coherence"""
    n1, n2 = C1.shape[0], C2.shape[0]
    cohe1 = pos_A[C1, :][:, C1].sum() / (n1 * (n1-1))
    cohe2 = pos_A[C2, :][:, C2].sum() / (n2 * (n2-1))
    return np.mean([cohe1, cohe2])


def oppo(neg_A, C1, C2):
    """opposition"""
    return neg_A[C1, :][:, C2].sum() / C1.shape[0] / C2.shape[0]


def agreement(pos_A, neg_A, C1, C2):
    """edge agreement ratio"""
    n_bad_pos = pos_A[C1, :][:, C2].sum() * 2
    n_bad_neg = neg_A[C1, :][:, C1].sum() + neg_A[C2, :][:, C2].sum()
    C = list(C1) + list(C2)
    n_total = pos_A[C, :][:, C].sum() + neg_A[C, :][:, C].sum()
    return 1 - (n_bad_pos + n_bad_neg) / n_total


def pc(A, C1, C2):
    """x \times A \times x' / (x \times x')"""
    x = np.zeros(A.shape[0])
    x[C1] = 1
    x[C2] = -1
    xT = x[:, None]
    return (x @ A @ xT / (x @ xT))[0]


def populate_fields(df, pos_A, neg_A, make_assertion=True):
    """use this!!!"""
    cols = df.columns
    existing_fields = ('C1', 'C2')
    for f in existing_fields:
        assert f in cols

    print('calculating degree-related')
    df['posdeg1'] = df['C1'].apply(lambda v: avg_pos_deg(pos_A, v))
    df['posdeg2'] = df['C2'].apply(lambda v: avg_pos_deg(pos_A, v))
    df['negdeg1'] = df['C1'].apply(lambda v: avg_neg_deg(neg_A, v))
    df['negdeg2'] = df['C2'].apply(lambda v: avg_neg_deg(neg_A, v))

    df['pos_avg'] = ((df['posdeg1'] * df['size1'] + df['posdeg1'] * df['size2'])
                     / (df['size1'] + df['size2']))
    
    df['max_posdeg'] = np.maximum(df['posdeg1'], df['posdeg2'])

    print('calculating ham')
    df['coh'] = df[['C1', 'C2']].apply(lambda d: cohe(pos_A, d['C1'], d['C2']), axis=1)
    df['opp'] = df[['C1', 'C2']].apply(lambda d: oppo(neg_A, d['C1'], d['C2']), axis=1)
    df['ham'] = 2 * df['coh'] * df['opp'] / (df['coh'] + df['opp'])

    print('calculating agreement')
    df['agreement'] = df[['C1', 'C2']].apply(lambda d: agreement(pos_A, neg_A, d['C1'], d['C2']), axis=1)

    if 'beta' not in cols and 'best_beta' not in cols:
        print('calculating beta')
        df['beta'] = df[['C1', 'C2']].apply(lambda d: sbr(pos_A - neg_A, d['C1'], d['C2']), axis=1)

    print('calculating PC')
    df['pc'] = df[['C1', 'C2']].apply(lambda d: pc(pos_A - neg_A, d['C1'], d['C2']), axis=1)

    idx = (df['coh'] != 0) & (df['opp'] != 0)

    if make_assertion:
        if idx.sum() > 0:
            print('filtering opp|coh=0 rows')
            df = df[idx].reindex()

        assert (df['coh'] >= 0).all()
        assert (df['coh'] <= 1).all()
        assert (df['opp'] >= 0).all()
        assert (df['opp'] <= 1).all(), (df['opp'] > 1).nonzero()
        assert (df['ham'] >= 0).all(), np.isnan(df['ham']).nonzero()
        assert (df['ham'] <= 1).all()
        assert (df['agreement'] >= 0).all()
        assert (df['agreement'] <= 1).all()

    return df
