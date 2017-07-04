import numpy as np
import torch
from layers import se3_to_rt
import ipdb


if __name__ == "__main__":
    se3dim = 7
    se3type = 'se3quat'
    input_tensor = np.zeros((2, 2, se3dim))
    input_tensor[0, 0, :] = [0.2269, 0.6909, 0.5513, 0.7192, 0.7195,
                             0.4911, 0.4231]
    input_tensor[0, 1, :] = [0.7800, 0.9808, 0.4109, 0.6848, 0.5797,
                             0.4809, 0.1400]

    input_tensor[1, 0, :] = [0.3921, 0.4010, 0.3432, 0.6273, 0.7290,
                             0.3242, 0.4386]
    input_tensor[1, 1, :] = [0.2448, 0.0597, 0.6948, 0.3980, 0.5939,
                             0.7380, 0.6318]

    features = torch.autograd.Variable(torch.Tensor(input_tensor),
                                       requires_grad=True)

    se3layer = se3_to_rt.SE3ToRt(se3type)
    output = se3layer.forward(features)

    print "input:"
    print features

    print "done, output:"
    print output
