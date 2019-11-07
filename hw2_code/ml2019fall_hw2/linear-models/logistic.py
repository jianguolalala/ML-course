import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
     # begin answer
    X = np.concatenate([np.ones((1,N)),X],axis=0)
    y[y<0]=0
    iters = 0
    alpha = 1.0
    while True:
        delta = np.dot(X, 1/(1+np.exp(-np.dot(X.T,w))) - y.T)/N
        w = w - alpha*delta
        pred = 1/(1+np.exp(-np.dot(w.T,X)))
        pred[pred>0.5] = 1
        pred[pred<0.5] = 0
        
        # print(np.sum(pred==y))
        if np.sum(pred==y) == N or iters>3000:
            break
        iters += 1
        alpha = np.power(0.9, iters//100)
    y[y==0]=-1
    # end answer
    
    return w, iters
