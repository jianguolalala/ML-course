import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer

    N_test, P = x.shape
    N, P = x_train.shape
    y = np.zeros(N_test)

    for i in range(N_test):
        distance = np.linalg.norm(x_train - x[i], axis=1)
        idx = np.argsort(distance)[:k]
        mode, count = scipy.stats.mode(y_train[idx])
        y[i] = mode
    # end answer

    return y
