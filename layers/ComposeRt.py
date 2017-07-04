import torch
from torch.autograd import Function
from torch.nn import Module

## FWD/BWD pass function
class ComposeRtFunction(Function):
	def __init__(self, rightToLeft = False):
		self.rightToLeft = rightToLeft;

	def forwardPair(self, A, B):
		# Check dimensions
		batch_size, num_rows, num_cols = A.size()
		assert (num_rows == 3 and num_cols == 4);
		assert (A.is_same_size(B));

		# Init for FWD pass
		rA = A.narrow(2, 0, 3);
		rB = B.narrow(2, 0, 3);
		tA = A.narrow(2, 3, 1);
		tB = B.narrow(2, 3, 1);

		# Init output
		output = A.new().resize_as_(A);
		output.narrow(2, 0, 3).copy_(torch.bmm(rA, rB));
		output.narrow(2, 3, 1).copy_(torch.baddbmm(tA, rA, tB));

		# Return
		return output;

	def backwardPair(self, grad, A, B):
		# Setup vars
		rA = A.narrow(2, 0, 3);
		rB = B.narrow(2, 0, 3);
		tA = A.narrow(2, 3, 1);
		tB = B.narrow(2, 3, 1);
		r_g = grad.narrow(2, 0, 3);
		t_g = grad.narrow(2, 3, 1);

		# Initialize grad input
		A_g = grad.new().resize_as_(grad);
		B_g = grad.new().resize_as_(grad);

		# Compute gradients w.r.t translations tA & tB
		# t = rA * tB + tA
		A_g.narrow(2, 3, 1).copy_(t_g);  # tA_g = t_g
		B_g.narrow(2, 3, 1).copy_(torch.bmm(rA.transpose(1, 2), t_g));  # tB_g = rA ^ T * t_g

		# Compute gradients w.r.t rotations rA & rB
		# r = rA * rB
		A_g.narrow(2, 0, 3).copy_(torch.bmm(r_g, rB.transpose(1, 2)).baddbmm_(t_g, tB.transpose(1,2)));  # rA_g = r_g * rB ^ T + (t_g * tB ^ T)
		B_g.narrow(2, 0, 3).copy_(torch.bmm(rA.transpose(1, 2), r_g));  # rB_g = rA ^ T * r_g (similar to translation grad, but with 3 column vectors)

		# Return
		return A_g, B_g;

	def forward(self, input):
		# Check dimensions
		batch_size, num_se3, num_rows, num_cols = input.size()
		assert (num_rows == 3 and num_cols == 4);

		# Compute output
		output = input.clone()
		for n in xrange(1,num_se3): # 1,2,3,...nSE3-1
			if self.rightToLeft: # Append to left
				output.select(1,n).copy_(self.forwardPair(input.select(1,n), output.select(1,n-1))) # T'_n = T_n * T'_n-1
			else: # Append to right
				output.select(1,n).copy_(self.forwardPair(output.select(1,n-1), input.select(1,n))) # T'_n = T'_n-1 * T_n
		self.save_for_backward(input, output)
		return output

	def backward(self, grad_output):
		# Get input and check for dimensions
		input, output = self.saved_tensors
		batch_size, num_se3, num_rows, num_cols = input.size()
		assert (num_rows == 3 and num_cols == 4);

		# Temp memory for gradient computation
		temp = grad_output.clone()

		# Compute gradient w.r.t input
		grad_input = input.clone().zero_()
		for n in xrange(num_se3-1,0,-1): # nSE3-1,...2,1
			if self.rightToLeft:
				A_g, B_g = self.backwardPair(temp.select(1,n), input.select(1,n), output.select(1,n-1)) # T'_n = T_n * T'_n-1
				grad_input.select(1,n).copy_(A_g) # Finished for this transform
				temp.select(1,n-1).add_(B_g)	  # Gradient w.r.t previous transform (added to it's grad_output)
			else:
				A_g, B_g = self.backwardPair(temp.select(1,n), output.select(1,n-1), input.select(1,n)) # T'_n = T'_n-1 * T_n
				grad_input.select(1,n).copy_(B_g) # Finished for this transform
				temp.select(1,n-1).add_(A_g)	  # Gradient w.r.t previous transform (added to it's grad_output)
		grad_input.select(1,0).copy_(temp.select(1,0))

		return grad_input

## FWD/BWD pass module
class ComposeRt(Module):
	def __init__(self, rightToLeft = False):
		super(ComposeRt, self).__init__()
		self.rightToLeft = rightToLeft

	def forward(self, input):
		return ComposeRtFunction(self.rightToLeft)(input)