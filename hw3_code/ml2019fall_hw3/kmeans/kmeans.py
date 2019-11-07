import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer

    iter_num = 100
    n, p = x.shape
    idx = np.zeros(n, dtype=np.int32)
    iter_ctrs = np.zeros((iter_num, k, p))
    init_ctrs_idx = np.random.choice(n, k, replace=False)
    iter_ctrs[0] = x[init_ctrs_idx]

    # print(iter_ctrs[0])
    # find best ctrs
    finalIter = 0
    for iter in range(1, iter_num):
        for i in range(n):
            t_x = x[i].reshape(1, -1)
            distance = np.linalg.norm(iter_ctrs[iter-1] - t_x, axis=1)
            t_idx = np.argwhere(distance == np.min(distance))[0][0]
            idx[i] = t_idx
        for i in range(k):
            # print('update', np.sum(x[idx == i], axis=0), np.sum(idx == i))
            iter_ctrs[iter,i] = np.sum(x[idx == i], axis=0) / (np.sum(idx == i) + 1)
        # print(iter_ctrs[iter])
        diff = iter_ctrs[iter] - iter_ctrs[iter - 1]
        if np.all(abs(diff) < 1e-6):
            finalIter = iter
            break
    ctrs = iter_ctrs[finalIter - 1, :, :]
    iter_ctrs = iter_ctrs[0: finalIter - 1, :, :]
    # end answer

    return idx, ctrs, iter_ctrs
