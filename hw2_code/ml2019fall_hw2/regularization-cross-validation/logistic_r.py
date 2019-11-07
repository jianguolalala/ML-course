import numpy as np
from scipy.special import expit

# expit is used to cal 1/(1+exp^(-x)) but have no overflow
def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    X = np.concatenate([np.ones((1,N)),X],axis=0)
    y[y<0]=0
    iters = 0
    alpha = 0.001
    while True:
        delta = (np.dot(X, expit(np.dot(X.T,w)) - y.T) + 2*lmbda*w)/N
        w = w - alpha*delta
        pred = expit(np.dot(X.T,w))
        pred[pred>0.5] = 1
        pred[pred<0.5] = 0
        
        # print(np.sum(pred==y))
        if np.sum(pred==y) == N or iters>100:
            break
        iters += 1
    y[y==0]=-1
    # end answer
    return w
