import torch
from torch.autograd import Function
import torch.nn.modules.loss as Loss
from _ext import se3layers

'''
	--------------------- Non-Rigidly deform the input point cloud by transforming and blending across multiple SE3s & compare to target ------------------------------
    Weighted3DTransformNormLoss() :
    Weighted3DTransformNormLoss.forward(inputpts, inputmasks, inputtfms, targetflows)
    Weighted3DTransformNormLoss.backward(grad_output)

    Given: 3D points (B x 3 x N x M), Masks (B x k x N x M) and Transforms (B x k x 3 x 4) 
    and a target flow cloud (B x 3 x N x M), this loss computes the following:
        Input  ==> pt: x_ij (j^th point in batch i), mask: w_ij (k-channel mask for j^th point in batch i), tfm: t_i (k transforms for batch i)
        Target ==> flow: f_ij (flow for j^th point in batch i)
        Loss   ==> L_ij = 0.5 * sum_k w_ij^k ( ((R_i^k * x_ij + t_i^k) - x_ij) - f_ij )^2 / sigma(f_ij)
    Ideally, the mask weights have to >= 0 and <= 1 and sum to 1 across the "k" channels.
    Essentially, we transform each point by all "k" transforms, compute the predicted flows, compute difference to target flows
    and normalize by a scalar proportional to the target flows to get rid of the scale dependence of the loss. We then
    weight the error for each of the "k" channels by the corresponding mask value "w". Ideally, if the mask is binary, each point will have an error
    only for the corresponding mask. If not, we will still measure error to all transforms but we weight it, so the network has an incentive
    to make these weights closer to binary so that each transform/mask only has to represent the motion of a subset of points.
'''

## FWD/BWD pass function
class Weighted3DTransformNormLossFunction(Function):
    def __init__(self, size_average=True, norm_wt=0.5, norm_per_pt=False):
        super(Weighted3DTransformNormLossFunction, self).__init__()
        self.size_average = size_average
        self.norm_wt      = norm_wt
        self.norm_per_pt  = norm_per_pt

    def forward(self, inputpts, inputmasks, inputtfms, targetflows):
        # Check dimensions
        batch_size, num_channels, data_height, data_width = inputpts.size()
        num_se3 = inputmasks.size()[1]
        assert (num_channels == 3);
        assert (inputmasks.size() == torch.Size([batch_size, num_se3, data_height, data_width]));
        assert (inputtfms.size() == torch.Size([batch_size, num_se3, 3, 4]));  # Transforms [R|t]

        # Run the FWD pass
        self.save_for_backward(inputpts, inputmasks, inputtfms, targetflows)  # Save for BWD pass
        if inputpts.is_cuda:
            output = se3layers.Weighted3DTransformNormLoss_forward_cuda(inputpts, inputmasks, inputtfms, targetflows,
                                                                        self.norm_wt, self.norm_per_pt, self.size_average)
        elif inputpts.type() == 'torch.DoubleTensor':
            output = se3layers.Weighted3DTransformNormLoss_forward_double(inputpts, inputmasks, inputtfms, targetflows,
                                                                          self.norm_wt, self.norm_per_pt, self.size_average)
        else:
            output = se3layers.Weighted3DTransformNormLoss_forward_float(inputpts, inputmasks, inputtfms, targetflows,
                                                                         self.norm_wt, self.norm_per_pt, self.size_average)

        # Return a tensor
        return inputpts.new((output,))

    def backward(self, grad_output):
        # Get saved tensors
        inputpts, inputmasks, inputtfms, targetflows = self.saved_tensors

        # Initialize grad input
        grad_inputpts = inputpts.new().resize_as_(inputpts)
        grad_inputmasks = inputmasks.new().resize_as_(inputmasks)
        grad_inputtfms = inputtfms.new().resize_as_(inputtfms)

        # Run the BWD pass
        if grad_output.is_cuda:
            se3layers.Weighted3DTransformNormLoss_backward_cuda(inputpts, inputmasks, inputtfms, targetflows,
                                                                grad_inputpts, grad_inputmasks, grad_inputtfms,
                                                                grad_output, self.norm_wt, self.norm_per_pt,
                                                                self.size_average)
        elif grad_output.type() == 'torch.DoubleTensor':
            se3layers.Weighted3DTransformNormLoss_backward_double(inputpts, inputmasks, inputtfms, targetflows,
                                                                  grad_inputpts, grad_inputmasks, grad_inputtfms,
                                                                  grad_output, self.norm_wt, self.norm_per_pt,
                                                                  self.size_average)
        else:
            se3layers.Weighted3DTransformNormLoss_backward_float(inputpts, inputmasks, inputtfms, targetflows,
                                                                 grad_inputpts, grad_inputmasks, grad_inputtfms,
                                                                 grad_output, self.norm_wt, self.norm_per_pt,
                                                                 self.size_average)

        # Return (no gradient for target)
        return grad_inputpts, grad_inputmasks, grad_inputtfms, None


## FWD/BWD pass module
class Weighted3DTransformNormLoss(torch.nn.Module):
    def __init__(self, size_average=True, norm_wt=0.5, norm_per_pt=False):
        super(Weighted3DTransformNormLoss, self).__init__()
        self.size_average = size_average
        self.norm_wt      = norm_wt
        self.norm_per_pt  = norm_per_pt

    def forward(self, inputpts, inputmasks, inputtfms, targetflows):
        Loss._assert_no_grad(targetflows)
        return Weighted3DTransformNormLossFunction(self.size_average, self.norm_wt,
                                                   self.norm_per_pt)(inputpts, inputmasks,inputtfms, targetflows)