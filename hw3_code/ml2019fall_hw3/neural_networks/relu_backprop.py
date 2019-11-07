import numpy as np

def relu_backprop(next_sensitivity, in_):
    '''
    The backpropagation process of relu
      input paramter:
          next_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: same as in_sensitivity
      
      output paramter:
          out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity
    '''
    # TODO

    # begin answer
    deri = np.where(in_ > 0, 1.0, 0.0)
    out_sensitivity = deri * next_sensitivity
    # end answer
    return out_sensitivity

