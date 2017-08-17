import torch
from torch.autograd import Function
import torch.nn.modules.loss as Loss
from _ext import se3layers

'''
	--------------------- Non-Rigidly deform the input point cloud by transforming and blending across multiple SE3s & compare to target ------------------------------
    Weighted3DTransformLoss() :
    Weighted3DTransformLoss.forward(inputpts, inputmasks, inputtfms, targetpts)
    Weighted3DTransformLoss.backward(grad_output)
   
    Given: 3D points (B x 3 x N x M), Masks (B x k x N x M) and Transforms (B x k x 3 x 4) 
    and a target point point cloud (B x 3 x N x M), this loss computes the following:
        Input  ==> pt: x_ij (j^th point in batch i), mask: w_ij (k-channel mask for j^th point in batch i), tfm: t_i (k transforms for batch i)
        Target ==> pt: y_ij (j^th point in batch i)
        Loss   ==> L_ij = 0.5 * sum_k w_ij^k || (R_i^k * x_ij + t_i^k) - y_ij ||^2
    Ideally, the mask weights have to >= 0 and <= 1 and sum to 1 across the "k" channels.
    Essentially, we transform each point by all "k" transforms, compute the difference between the transformed points and the GT target point
    and weight the error for each of the "k" channels by the corresponding mask value "w". Ideally, if the mask is binary, each point will have an error
    only for the corresponding mask. If not, we will still measure error to all transforms but we weight it by the weight, so the network has an incentive
    to make these weights closer to binary so that each transform/mask only has to represent the motion of a subset of points.
'''

## FWD/BWD pass function
class Weighted3DTransformLossFunction(Function):
    def __init__(self, size_average=True):
        super(Weighted3DTransformLossFunction, self).__init__()
        self.size_average = size_average

    def forward(self, inputpts, inputmasks, inputtfms, targetpts):
        # Check dimensions
        batch_size, num_channels, data_height, data_width = inputpts.size()
        num_se3 = inputmasks.size()[1]
        assert (num_channels == 3);
        assert (inputmasks.size() == torch.Size([batch_size, num_se3, data_height, data_width]));
        assert (inputtfms.size() == torch.Size([batch_size, num_se3, 3, 4]));  # Transforms [R|t]

        # Run the FWD pass
        self.save_for_backward(inputpts, inputmasks, inputtfms, targetpts)  # Save for BWD pass
        if inputpts.is_cuda:
            output = se3layers.Weighted3DTransformLoss_forward_cuda(inputpts, inputmasks, inputtfms, targetpts, self.size_average)
        elif inputpts.type() == 'torch.DoubleTensor':
            output = se3layers.Weighted3DTransformLoss_forward_double(inputpts, inputmasks, inputtfms, targetpts, self.size_average)
        else:
            output = se3layers.Weighted3DTransformLoss_forward_float(inputpts, inputmasks, inputtfms, targetpts, self.size_average)

        # Return a tensor
        return inputpts.new((output,))

    def backward(self, grad_output):
        # Get saved tensors
        inputpts, inputmasks, inputtfms, targetpts = self.saved_tensors

        # Initialize grad input
        grad_inputpts   = inputpts.new().resize_as_(inputpts)
        grad_inputmasks = inputmasks.new().resize_as_(inputmasks)
        grad_inputtfms  = inputtfms.new().resize_as_(inputtfms)

        # Run the BWD pass
        if grad_output.is_cuda:
            se3layers.Weighted3DTransformLoss_backward_cuda(inputpts, inputmasks, inputtfms, targetpts,
                                                            grad_inputpts, grad_inputmasks, grad_inputtfms,
                                                            grad_output, self.size_average)
        elif grad_output.type() == 'torch.DoubleTensor':
            se3layers.Weighted3DTransformLoss_backward_double(inputpts, inputmasks, inputtfms, targetpts,
                                                              grad_inputpts, grad_inputmasks, grad_inputtfms,
                                                              grad_output, self.size_average)
        else:
            se3layers.Weighted3DTransformLoss_backward_float(inputpts, inputmasks, inputtfms, targetpts,
                                                             grad_inputpts, grad_inputmasks, grad_inputtfms,
                                                             grad_output, self.size_average)

        # Return (no gradient for target)
        return grad_inputpts, grad_inputmasks, grad_inputtfms, None


## FWD/BWD pass module
class Weighted3DTransformLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(Weighted3DTransformLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputpts, inputmasks, inputtfms, targetpts):
        Loss._assert_no_grad(targetpts)
        return Weighted3DTransformLossFunction(self.size_average)(inputpts, inputmasks, inputtfms, targetpts)
