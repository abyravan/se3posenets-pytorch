import torch
from torch.autograd import Function
from torch.nn import Module
from _ext import se3layers

## FWD/BWD pass function
class NTfm3DFunction(Function):
	def __init__(self):
		self.output 	 = None

	def forward(self, points, masks, transforms):
		# Check dimensions
		batch_size, num_channels, data_height, data_width = points.size()
		num_se3 = masks.size()[1]
		assert(num_channels == 3);
		assert(masks.size() == torch.Size([batch_size, num_se3, data_height, data_width])); 
		assert(transforms.size() == torch.Size([batch_size, num_se3, 3, 4])); # Transforms [R|t]

		# Create output (or reshape)
		if not self.output:
			self.output = points.clone().zero_();
		elif not self.output.is_same_size(points):
			self.output.resize_as_(points);
		
		# Run the FWD pass
		self.save_for_backward(points, masks, transforms); # Save for FWD pass
		if not points.is_cuda:
			se3layers.NTfm3D_forward_float(points, masks, transforms, self.output);
		else:
			se3layers.NTfm3D_forward_cuda(points, masks, transforms, self.output);
		
		# Return
		return self.output

	def backward(self, grad_output):
		# Get saved tensors
		points, masks, transforms = self.saved_tensors
		assert(grad_output.is_same_size(self.output));
		
		# Initialize grad input
		grad_points 	 = points.new().resize_as_(points)
		grad_masks 		 = masks.new().resize_as_(masks)
		grad_transforms = transforms.new().resize_as_(transforms)
		
		# Run the BWD pass
		if not grad_output.is_cuda:
			se3layers.NTfm3D_backward_float(points, masks, transforms, self.output,
													  grad_points, grad_masks, grad_transforms, grad_output);
		else:
			se3layers.NTfm3D_backward_cuda(points, masks, transforms, self.output,
													 grad_points, grad_masks, grad_transforms, grad_output);
		
		# Return
		return grad_points, grad_masks, grad_transforms;

## FWD/BWD pass module
class NTfm3D(Module):
	def __init__(self):
		super(NTfm3D, self).__init__()

	def forward(self, points, masks, transforms):
		return NTfm3DFunction()(points, masks, transforms)
