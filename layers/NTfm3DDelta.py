import torch
from torch.autograd import Function
from torch.nn import Module
from _ext import se3layers

'''
	--------------------- Non-Rigidly deform the input point cloud by transforming and blending across multiple SE3s -------------------------------
	-- This computes a weighted avg. of the deltas produced by each transforms by the masks and adds it to the point cloud to make the prediction --
   NTfm3DDelta() :
   NTfm3DDelta.forward(3D points, masks, Rt)
   NTfm3DDelta.backward(grad_output)

   NTfm3DDelta will transform the given input points "x" (B x 3 x N x M) and "k" masks (B x k x N x M) via a set of 3D affine transforms (B x k x 3 x 4), 
	resulting in a set of transformed 3D points (B x 3 x N x M). The transforms will be applied to all the points 
	and their outputs are interpolated based on the mask weights:
		output = x + mask(1,...) .* (R_1 * x + t_1 - x) + mask(2,...) .* (R_2 * x + t_2 - x) + .....
	Each 3D transform is a (3x4) matrix [R|t], where "R" is a (3x3) affine matrix and "t" is the translation (3x1).
	Note: The mask values have to sum to 1.0
		sum(mask,2) = 1
'''


## FWD/BWD pass function
class NTfm3DDeltaFunction(Function):
    def __init__(self):
        self.output = None

    def forward(self, points, masks, transforms):
        # Check dimensions
        batch_size, num_channels, data_height, data_width = points.size()
        num_se3 = masks.size()[1]
        assert (num_channels == 3);
        assert (masks.size() == torch.Size([batch_size, num_se3, data_height, data_width]));
        assert (transforms.size() == torch.Size([batch_size, num_se3, 3, 4]));  # Transforms [R|t]

        # Create output (or reshape)
        if not self.output:
            self.output = points.clone().zero_();
        elif not self.output.is_same_size(points):
            self.output.resize_as_(points);

        # Run the FWD pass
        self.save_for_backward(points, masks, transforms);  # Save for BWD pass
        if points.is_cuda:
            se3layers.NTfm3DDelta_forward_cuda(points, masks, transforms, self.output);
        elif points.type() == 'torch.DoubleTensor':
            se3layers.NTfm3DDelta_forward_double(points, masks, transforms, self.output);
        else:
            se3layers.NTfm3DDelta_forward_float(points, masks, transforms, self.output);

        # Return
        return self.output

    def backward(self, grad_output):
        # Get saved tensors
        points, masks, transforms = self.saved_tensors
        assert (grad_output.is_same_size(self.output));

        # Initialize grad input
        grad_points = points.new().resize_as_(points)
        grad_masks = masks.new().resize_as_(masks)
        grad_transforms = transforms.new().resize_as_(transforms)

        # Run the BWD pass
        if grad_output.is_cuda:
            se3layers.NTfm3DDelta_backward_cuda(points, masks, transforms, self.output,
                                           grad_points, grad_masks, grad_transforms, grad_output);
        elif grad_output.type() == 'torch.DoubleTensor':
            se3layers.NTfm3DDelta_backward_double(points, masks, transforms, self.output,
                                             grad_points, grad_masks, grad_transforms, grad_output);
        else:
            se3layers.NTfm3DDelta_backward_float(points, masks, transforms, self.output,
                                            grad_points, grad_masks, grad_transforms, grad_output);

        # Return
        return grad_points, grad_masks, grad_transforms;


## FWD/BWD pass module
class NTfm3DDelta(Module):
    def __init__(self):
        super(NTfm3DDelta, self).__init__()

    def forward(self, points, masks, transforms):
        return NTfm3DDeltaFunction()(points, masks, transforms)
