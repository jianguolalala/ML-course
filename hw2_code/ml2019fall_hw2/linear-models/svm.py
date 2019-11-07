import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros(P + 1)
    num = 0
    X = np.concatenate([np.ones((1,N)),X],axis=0)

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    def loss(w):
        return np.sum(w*w)/2
    def cal(w):
        w = w.reshape((-1, 1))
        res = y*(np.dot(w.T, X))
        return res.reshape(-1)
    
    cons = NonlinearConstraint(cal, lb=1, ub=np.inf)
    res = minimize(loss, w, constraints=cons)
    # end answer
    return res.x.reshape((-1, 1)), res.nit

