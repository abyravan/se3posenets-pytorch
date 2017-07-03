import torch
from torch.autograd import Function


def check(in, gradOut):
    return


class Se3ToRtFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.output = None
        self.feature_size = None

    def forward(self, features):
        batch_size, numSE3s, nparams = features.size()

        totSE3 = batch_size * numSE3s

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        return grad_input, None


class Se3ToRt(torch.nn.Module):
    def __init__(self, transform_type):
        super(Se3ToRt, self).__init__()
        self.transform_type = transform_type

    def forward(self, features):
        return Se3ToRtFunction(transform_type)(features)
        # return Se3ToRtFunction(spatial_scale)(features, rois)
