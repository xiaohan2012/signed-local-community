import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs

from sklearn.cluster import KMeans

from helpers import (
    signed_laplacian,
    signed_normalized_laplacian
)


def signed_spectral_clustering(g, k, normalize=False, weight='sign'):
    if normalize:
        A = nx.adj_matrix(g, weight=weight)
        L = signed_normalized_laplacian(A)
    else:
        L = signed_laplacian(g)

    eig_vals, eig_vects = eigs(L.asfptype(), k=k, which='SM')
    eig_vects = eig_vects[:, np.argsort(eig_vals)]  # sort them by eigen values
    eig_vects = np.real(eig_vects)

    embedding = eig_vects[:, :k-1]  # top-(k-1) eigen vectors
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embedding)
    return kmeans.labels_
                

