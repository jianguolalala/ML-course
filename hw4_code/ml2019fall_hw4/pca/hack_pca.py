import numpy as np
from PIL import Image
from pca import PCA


def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    # YOUR CODE HERE
    # begin answer
    img = np.asarray(Image.open(filename).convert('L'))
    x_idx, y_idx = np.where(img < 30)
    input_data = np.hstack((x_idx.reshape(-1, 1), y_idx.reshape(-1, 1)))
    mean = np.mean(input_data, axis=0)
    input_data = input_data - mean
    eigvector, eigvalue = PCA(input_data)
    vec = eigvector[:, 0]
    angle = np.arctan(vec[0]/vec[1]) * 180 / np.pi
    final_image = Image.open(filename).rotate(-angle)
    return final_image, angle