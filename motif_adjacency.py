def motif_eef(A):
    """motif: enemy - enemy - friend"""
    # path a ---(+)--- b ---(-)--- c
    C1 = (A @ A)
    C1[C1 > 0] = 0

    C2 = A.copy()
    C2[C2 > 0] = 0
    pos_neg_C = C1 * C2

    # path a ---(-)--- b ---(-)--- c
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    C1 = (neg_A @ neg_A)
    C1[C1 < 0] = 0

    C2 = A.copy()
    C2[C2 < 0] = 0

    neg_neg_C = C1 * C2
    W = pos_neg_C + neg_neg_C
    return W
