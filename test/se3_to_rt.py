import torch
from torch.autograd import Function, Variable
import ipdb


# class SE3ToRtFunction(Function):
#     def __init__(self, transform_type):
#         self.output = None
#         self.feature_size = None
#         self.transform_type = transform_type

#     def forward(self, features):
#         self.feature_size = features.size()
#         batch_size, numSE3s, nparams = self.feature_size

#         totSE3 = batch_size * numSE3s
#         ncols = 4
#         output = torch.zeros((batch_size, numSE3s, 3, ncols))

#         if self.transform_type == 'se3quat':
#             rotdim = 4
#         elif self.transform_type == 'affine':
#             rotdim = 9
#         else:
#             rotdim = 3

#         if self.transform_type == 'affine':
#             output.narrow(3, 3, 1).copy_(features.narrow(2, 0, 3),
#                                          broadcast=True)
#             output.narrow(3, 0, 3).copy_(features.narrow(2, 3, rotdim))
#             return output

#         if self.transform_type == 'se3quat':
#             params = features.view(totSE3, -1)
#             quat = params.narrow(1, 3, rotdim)
#             self.unitquat = self._computeUnitQuaternionFromQuaternion(quat)
#             rot = output.view(totSE3, 3, ncols).narrow(2, 0, 3)
#             rot.copy_(self._createRotFromUnitQuaternion(self.unitquat))

#         transparams = params.narrow(1, 0, 3)
#         output.narrow(3, 3, 1).copy_(transparams)

#         return output

#     def backward(self, grad_output):
#         assert(self.feature_size is not None)

#         # batch_size, numSE3s, nparams = self.feature_size
#         return

class SE3ToRt(torch.nn.Module):
    def __init__(self, transform_type):
        super(SE3ToRt, self).__init__()
        self.transform_type = transform_type

    def _computeUnitQuaternionFromQuaternion(self, quat):
        norm2 = torch.pow(quat, 2).sum(1)
        norm = norm2.sqrt()

        unitquat = torch.div(quat, norm.expand_as(quat))

        # post-processing
        # threshold = 1e-12
        # N = quat.size(0)
        # for n in range(N):
        #     if norm2[n].data[0] < threshold:
        #         unitquat[n, 0:3] = 0.0
        #         unitquat[n, 3] = 1.0

        return unitquat

    def _createRotFromUnitQuaternion(self, unitquat):

        N = unitquat.size(0)
        rot = Variable(torch.Tensor(N, 3, 3).type_as(unitquat.data))

        x = unitquat.narrow(1, 0, 1)
        y = unitquat.narrow(1, 1, 1)
        z = unitquat.narrow(1, 2, 1)
        w = unitquat.narrow(1, 3, 1)

        x2 = torch.pow(x, 2)
        y2 = torch.pow(y, 2)
        z2 = torch.pow(z, 2)
        w2 = torch.pow(w, 2)

        # row 1
        rot[:, 0, 0] = w2 + x2 - y2 - z2
        rot[:, 0, 1] = 2.0 * (torch.mul(x, y) - torch.mul(w, z))
        rot[:, 0, 2] = 2.0 * (torch.mul(x, z) + torch.mul(w, y))

        # row 2
        rot[:, 1, 0] = 2.0 * (torch.mul(x, y) + torch.mul(w, z))
        rot[:, 1, 1] = w2 - x2 + y2 - z2
        rot[:, 1, 2] = 2.0 * (torch.mul(y, z) - torch.mul(w, x))

        # row 3
        rot[:, 2, 0] = 2.0 * (torch.mul(x, z) - torch.mul(w, y))
        rot[:, 2, 1] = 2.0 * (torch.mul(y, z) + torch.mul(w, x))
        rot[:, 2, 2] = w2 - x2 - y2 + z2
        return rot

    def forward(self, features):
        self.feature_size = features.size()
        batch_size, numSE3s, nparams = self.feature_size

        totSE3 = batch_size * numSE3s
        ncols = 4

        if self.transform_type == 'se3quat':
            rotdim = 4
        elif self.transform_type == 'affine':
            rotdim = 9
        else:
            rotdim = 3

        # if self.transform_type == 'affine':
        #     output.narrow(3, 3, 1).copy_(features.narrow(2, 0, 3),
        #                                  broadcast=True)
        #     output.narrow(3, 0, 3).copy_(features.narrow(2, 3, rotdim))
        #     return output

        # output = Variable(torch.zeros((totSE3, 3, ncols)))
        if self.transform_type == 'se3quat':
            params = features.view(totSE3, -1)
            quat = params.narrow(1, 3, rotdim)
            self.unitquat = self._computeUnitQuaternionFromQuaternion(quat)
            rotations = self._createRotFromUnitQuaternion(
                self.unitquat)

        transparams = features.view(totSE3, -1, 1).narrow(1, 0, 3)

        output = torch.cat([rotations, transparams], dim=2).view(
            [batch_size, numSE3s, 3, ncols]).contiguous()

        return output

    # def forward(self, features):
        # return SE3ToRtFunction(self.transform_type)(features)
