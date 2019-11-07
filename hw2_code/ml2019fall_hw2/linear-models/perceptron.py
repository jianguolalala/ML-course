import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    X = np.concatenate([np.ones((1,N)),X], axis=0)
    P += 1
    alpha = 1.0
    # begin answer
    while True:
        dis = np.dot(w.T, X)*y
        error = np.argwhere(dis <= 0)
        if len(error)==0 or iters>1000:
            break
        w = w + alpha*np.sum((X[:,error[:,1]])*(y[:,error[:,1]]),axis=1,keepdims=True)
        alpha = np.power(0.9, iters//100)
        iters +=1
    # end answer
    
    return w, iters