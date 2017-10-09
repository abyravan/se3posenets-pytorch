import torch
from torch.autograd import Function
from torch.nn import Module

'''
   --------------------- Predicts a weighted average of points weighted by the "k" mask channels ------------------------------
   WeightedAveragePoints() :
   WeightedAveragePoints.forward(points, weights)
   WeightedAveragePoints:backward(points, weights)

   Given input points: (B x J x H x W) and "K" channel weights (B x K x H x W), this layer predicts a "K" 3D points (B x K x J) which are a 
	result of weighted averaging the points with the masks:
		p_i = sum_j (w_ij * x_j) / sum_j(w_ij) , i = 1 to k, j = 1 to N*M , x_j = [x1,x2,..xj] (a jD point) & p_i is the weighted averaged output.
'''

## FWD/BWD pass function
class WeightedAveragePointsFunction(Function):
	def forward(self, points, weights):
		# Check dimensions (B x J x H x W)
		B, J, H, W = points.size()
		K = weights.size(1)
		assert(weights.size(0) == B and weights.size(2) == H and weights.size(3) == W)

		# Compute output = B x K x J
		output = points.new().resize_(B, K, J)
		for i in range(K):
			M = weights.view(B,K,H*W).narrow(1,i,1).expand(B,J,H*W) # Get weights
			S = M.sum(2).clamp_(min=1e-12) # Clamp normalizing constant
			output.narrow(1,i,1).copy_((M * points).sum(2) / S) # Compute convex combination

		# Return
		self.save_for_backward(points, weights, output)
		return output

	def backward(self, grad_output):
		# Get saved tensors
		points, weights, output = self.saved_tensors
		B, J, H, W = points.size()
		K = weights.size(1)
		assert(grad_output.size(0) == B and grad_output.size(1) == K and grad_output.size(2) == J)

		# Compute grad points and weights
		grad_points  = points.new().resize_as_(points).zero_()
		grad_weights = weights.new().resize_as_(weights).zero_()
		for i in range(K):
			# Get the weights
			M = weights.view(B,K,H*W).narrow(1,i,1).expand(B,J,H*W)  # Get weights
			S = M.sum(2).clamp_(min=1e-12)  # Clamp normalizing constant

			# Get output gradients
			G = grad_output.view(B,K,J,1).narrow(1,i,1) / S # g_o / sum_j w_ij

			# Gradient w.r.t points
			grad_points += (M * G.expand(B,1,J,H*W).squeeze()) # g_o * ( w_ij / sum (w_ij) )

			# Gradients w.r.t wts (scale difference of each point from output by gradient & normalize by sum of wts)
			O = output.view(B,K,J,1).narrow(1,i,1).expand(B,1,J,H*W) # i^th output
			grad_weights.narrow(1,i,1).copy_(((points - O) * G.expand(B,1,J,H*W)).sum(1)) # (a_i - output) .* (gradOutput / sum) & sum across pt channels

		# Return
		return grad_points, grad_weights

## FWD/BWD pass module
class WeightedAveragePoints(Module):
	def __init__(self):
		super(WeightedAveragePoints, self).__init__()

	def forward(self, points, weights):
		# Run the rest of the FWD pass
		return WeightedAveragePointsFunction()(points, weights)