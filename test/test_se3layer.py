import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from layers import se3_to_rt

import ipdb


if __name__ == "__main__":
    se3dim = 7
    se3type = 'se3quat'
    torch.manual_seed(123)

    input_tensor = torch.rand(2, 2, se3dim)
    # input_tensor[0, 0] = 0.0

    features = torch.autograd.Variable(input_tensor,
                                       requires_grad=True)

    se3layer = se3_to_rt.SE3ToRt(se3type)
    output = se3layer.forward(features)
    target = Variable(torch.rand(output.size()))
    err = nn.MSELoss()(output, target)
    err.backward()
    gauto = features.grad.clone()

    print "input:"
    print features

    print "output:"
    print output

    print "grad"
    print gauto
