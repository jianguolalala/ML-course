import numpy as np

def fullyconnect_feedforward(in_, weight, bias):
    '''
    The feedward process of fullyconnect
      input parameters:
          in_     : the intputs, shape: [number of images, number of inputs]
          weight  : the weight matrix, shape: [number of inputs, number of outputs]
          bias    : the bias, shape: [number of outputs, 1]

      output parameters:
          out     : the output of this layer, shape: [number of images, number of outputs]
    '''
    # TODO

    # begin answer
    N, P = in_.shape
    in_ = np.hstack((np.ones((N,1)), in_))
    w = np.vstack((bias.T, weight))
    out = np.dot(in_, w)
    # end answer

    return out

