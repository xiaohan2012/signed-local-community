def motif_eef(A):
    """motif: enemy - enemy - friend
    assumes A is array
    """
    # path a ---(+)--- b ---(-)--- c
    C1 = (A @ A)
    C1[C1 > 0] = 0  # only keeps wedges (+- or -+)

    C2 = A.copy()
    C2[C2 > 0] = 0  # only - edges
    pos_neg_C = C1.multiply(C2)  # neg (u, v) s.t. there is an w (u, w) is + and (v, w) is -

    # path a ---(-)--- b ---(-)--- c
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0  # only - edges
    C1 = (neg_A @ neg_A)
    C1[C1 < 0] = 0  # only keeps -- wedges

    C2 = A.copy()
    C2[C2 < 0] = 0  # only + edges

    neg_neg_C = C1.multiply(C2)  # pos (u, v) s.t. there is an w that both (u, w) and (v, w) is -1
    W = pos_neg_C + neg_neg_C
    return W


def motif_eef_anchored(A):
    """
    captures the following motif with a and b as anchor nodes
         c
        /  \
       /    \
      -      -
     /        \
    [a]___+___[b]
    
    """
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0  # only - edges
    C1 = (neg_A @ neg_A)
    C1[C1 < 0] = 0  # only keeps -- wedges

    C2 = A.copy()
    C2[C2 < 0] = 0  # only + edges

    neg_neg_C = C1.multiply(C2)  # pos (u, v) s.t. there is an w that both (u, w) and (v, w) is -1
    return neg_neg_C
    

def motif_fff(A):
    """
    motif = friend-friend-friend triangles
    """
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0  # only + edges
    C1 = (pos_A @ pos_A)
    return C1.multiply(pos_A)


def motif_ff(A):
    """
    simply keep the + edges
    """
    pos_A = A.copy()
    pos_A[pos_A < 0] = 0  # only + edges
    return pos_A.multiply(pos_A)  # do not materialize the 0 entries    


M1, M2, M3 = 'm1', 'm2', "m3"
MOTIF2F = {
    M1: motif_eef_anchored,
    M2: motif_fff,
    M3: motif_ff
}
