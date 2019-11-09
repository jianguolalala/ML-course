import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    n = W.shape[0]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    Dm = np.diag(1. / np.sqrt(np.sum(W, axis=1)))
    L = np.dot(np.dot(Dm, L), Dm)
    w, v = np.linalg.eig(L)
    index = np.argsort(w)[0:k]
    v_k = v[:, index]
    v_k = v_k / np.linalg.norm(v_k, axis=1,keepdims=True)         
    idx = kmeans(v_k, k)
    return idx
    # end answer
