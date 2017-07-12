import torch
from torch.autograd import Function
from torch.nn import Module

'''
   --------------------- Converts SE3s to (R,t) ------------------------------
   SE3ToRt(transformType,hasPivot) :
   SE3ToRt:updateOutput(params)
   SE3ToRt:updateGradInput(params, gradOutput)

   SE3ToRt layer takes in [batchSize x nSE3 x p] values and converts them to [B x N x 3 x 4] or [B x N x 3 x 5] matrix where each transform has:
	"R", a 3x3 affine matrix and "t", a 3x1 translation vector and optionally, a 3x1 pivot vector.
	The layer takes SE3s in different forms and converts each to R" & "t" & "p" based on the type of transform. 
	Parameters are always shaped as: [trans, rot, pivot] where the pivot is optional.
	1) affine 	- Parameterized by 12 values (12-dim vector). "params" need to be 12 values which will be reshaped as a (3x4) matrix.
					  Input parameters are shaped as: [B x k x 12] matrix
	2) se3euler - SE3 transform with an euler angle parameterization for rotation.
					  "params" are 3 translational and 3 rotational (xyz-euler) parameters = [tx,ty,tz,r1,r2,r3].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
					  We take the triplet [r1,r2,r3], all in radians and compute a rotation matrix R which is decomposed as:
					  R = Rz(r3) * Ry(r2) * Rx(r1) where Rz,Ry,Rx are rotation matrices around z,y,x axes with angles of r3,r2,r1 respectively.
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 6] matrix	
	3) se3aa 	- SE3 transform with an axis-angle parameterization for rotation. 
					  "params" are 3 translational and 3 rotational (axis-angle) parameters = [tx,ty,tz,r1,r2,r3].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
							Rotation angle = || [r1,r2,r3] ||_2 which is the 2-norm of the vector
							Rotation axis  = [r1,r2,r3]/angle   which is the unit vector
					  		These are then converted to a rotation matrix (R) using the Rodriguez's transform
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 6] matrix	
	4) se3quat  - SE3 transform with a quaternion parameterization for the rotation.
					  "params" are 3 translational and 4 rotational (quaternion) parameters = [tx,ty,tz,qx,qy,qz,qw].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
							Unit Quaternion = [qx,qy,qz,qw] / || [qx,qy,qz,qw] ||_2  
					  		These are then converted to a rotation matrix (R) using the quaternion to rotation transform
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 7] matrix	
   5) se3spquat - SE3 transform with a stereographic projection of a quaternion as the parameterization for the rotation.
                 "params" are 3 translational and 3 rotational (SP-quaternion) parameters = [tx,ty,tz,sx,sy,sz].
					  The rotational parameters are converted to the rotation matrix form using the following parameterization:
							SP Quaternion -> Quaternion -> Unit Quaternion -> Rotation Matrix 
					  Translational parameters [tx,ty,tz] are the translations along the X,Y,Z axes in m	
					  Input parameters are shaped as: [B x k x 6] matrix	. For more details on this parameterization, check out: 
                 "A Recipe on the Parameterization of Rotation Matrices for Non-Linear Optimization using Quaternions" &
                 https://github.com/FugroRoames/Rotations.jl
	By default, transformtype is set to "affine"
'''

## FWD/BWD pass function
class SE3ToRtFunction(Function):
	def __init__(self, transform_type='affine', has_pivot=False):
		self.transform_type = transform_type
		self.has_pivot 		= has_pivot
		self.eps			= 1e-12

	## Check sizes
	def check(self, input, grad_output=None):
		# Input size
		batch_size, num_se3, num_params = input.size()
		num_pivot = 3 if (self.has_pivot) else 0
		if (self.transform_type == 'affine'):
			assert(num_params == 12+num_pivot)
		elif (self.transform_type == 'se3euler' or
			  self.transform_type == 'se3aa'    or
			  self.transform_type == 'se3spquat'):
			assert(num_params == 6+num_pivot)
		elif (self.transform_type == 'se3quat'):
			assert(num_params == 7+num_pivot)
		else:
			print("Unknown transform type input: {0}".format(self.transform_type))
			assert(False);

		# Gradient size
		if grad_output is not None:
			num_cols = 5 if (self.has_pivot) else 4
			assert (grad_output.size() == torch.Size([batch_size, num_se3, 3, num_cols]));

	########
	#### General helpers
	# Create a skew-symmetric matrix "S" of size [B x 3 x 3] (passed in) given a [B x 3] vector
	def create_skew_symmetric_matrix(self, vector):
		# Create the skew symmetric matrix:
		# [0 -z y; z 0 -x; -y x 0]
		N = vector.size(0)
		output = vector.new().resize_(N,3,3).fill_(0)
		output[:, 0, 1] = -vector[:, 2]
		output[:, 1, 0] =  vector[:, 2]
		output[:, 0, 2] =  vector[:, 1]
		output[:, 2, 0] = -vector[:, 1]
		output[:, 1, 2] = -vector[:, 0]
		output[:, 2, 1] =  vector[:, 0]
		return output

	########
	#### XYZ Euler representation helpers

	# Rotation about the X-axis by theta
	# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.7)
	def create_rot1(self, theta):
		N = theta.size(0)
		rot = torch.eye(3).view(1,3,3).expand(N,3,3).type_as(theta)
		rot[:, 1, 1] =  torch.cos(theta)
		rot[:, 2, 2] =  rot[:, 1, 1]
		rot[:, 1, 2] =  torch.sin(theta)
		rot[:, 2, 1] = -rot[:, 1, 2]
		return rot

	# Rotation about the Y-axis by theta
	# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
	def create_rot2(self, theta):
		N = theta.size(0)
		rot = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(theta)
		rot[:, 0, 0] =  torch.cos(theta)
		rot[:, 2, 2] =  rot[:, 0, 0]
		rot[:, 2, 0] =  torch.sin(theta)
		rot[:, 0, 2] = -rot[:, 2, 0]
		return rot

	# Rotation about the Z-axis by theta
	# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
	def create_rot3(self, theta):
		N = theta.size(0)
		rot = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(theta)
		rot[:, 0, 0] =  torch.cos(theta)
		rot[:, 1, 1] =  rot[:, 0, 0]
		rot[:, 0, 1] =  torch.sin(theta)
		rot[:, 1, 0] = -rot[:, 0, 1]
		return rot

	########
	#### Axis-Angle representation helpers

	# Compute the rotation matrix R from the axis-angle parameters using Rodriguez's formula:
	# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
	# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
	# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	def create_rot_from_aa(self, params):
		# Get the un-normalized axis and angle
		N = params.size(0)
		axis   = params.clone().view(N,3,1) 	# Un-normalized axis
		angle2 = (axis * axis).sum(1)			# Norm of vector (squared angle)
		angle  = torch.sqrt(angle2)				# Angle

		# Compute skew-symmetric matrix "K" from the axis of rotation
		K  = self.create_skew_symmetric_matrix(axis)
		K2 = torch.bmm(K, K)						 # K * K

		# Compute sines
		S  = torch.sin(angle) / angle
		S.masked_fill_(angle2.lt(self.eps), 1)  # sin(0)/0 ~= 1

		# Compute cosines
		C  = (1 - torch.cos(angle)) / angle2
		C.masked_fill_(angle2.lt(self.eps), 0)  # (1 - cos(0))/0^2 ~= 0

		# Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
		rot  = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(params) # R = I
		rot += K  * S.expand(N, 3, 3)									  # R = I + (sin(theta)/theta)*K
		rot += K2 * C.expand(N, 3, 3)									  # R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2
		return rot

	########
	#### Quaternion representation helpers

	# Compute the rotation matrix R from a set of unit-quaternions (N x 4):
	# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 9)
	def create_rot_from_unitquat(self, unitquat):
		# Init memory
		N 	= unitquat.size(0)
		rot = unitquat.new().resize_(N,3,3)

		# Get quaternion elements. Quat = [qx,qy,qz,qw] with the scalar at the rear
		x,y,z,w 	= unitquat[:,0], unitquat[:,1], unitquat[:,2], unitquat[:,3]
		x2,y2,z2,w2 = x*x, y*y, z*z, w*w

		# Row 1
		rot[:, 0, 0] = w2 + x2 - y2 - z2	# rot(0,0) = w^2 + x^2 - y^2 - z^2
		rot[:, 0, 1] = 2*(x*y - w*z)		# rot(0,1) = 2*x*y - 2*w*z
		rot[:, 0, 2] = 2*(x*z + w*y)		# rot(0,2) = 2*x*z + 2*w*y

		# Row 2
		rot[:, 1, 0] = 2*(x*y + w*z)		# rot(1,0) = 2*x*y + 2*w*z
		rot[:, 1, 1] = w2 - x2 + y2 - z2	# rot(1,1) = w^2 - x^2 + y^2 - z^2
		rot[:, 1, 2] = 2*(y*z - w*x)		# rot(1,2) = 2*y*z - 2*w*x

		# Row 3
		rot[:, 2, 0] = 2*(x*z - w*y)		# rot(2,0) = 2*x*z - 2*w*y
		rot[:, 2, 1] = 2*(y*z + w*x)		# rot(2,1) = 2*y*z + 2*w*x
		rot[:, 2, 2] = w2 - x2 - y2 + z2	# rot(2,2) = w^2 - x^2 - y^2 + z^2

		# Return
		return rot

	# Compute the unit quaternion from a quaternion
	def create_unitquat_from_quat(self, quat):
		# Compute the quaternion norms
		N = quat.size(0)
		norm2 = (quat * quat).sum(1) # Norm-squared
		norm  = torch.sqrt(norm2)    # Length of the quaternion

		# Compute the unit quaternion
		# TODO: No check for normalization issues currently
		unitquat = quat / (norm.expand_as(quat)) # Normalize the quaternion

		# Return
		return unitquat

	# Compute the derivatives of the rotation matrix w.r.t the unit quaternion
	# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 33-36)
	def compute_grad_rot_wrt_unitquat(self, unitquat):
		# Compute dR/dq' (9x4 matrix)
		N = unitquat.size(0)
		x,y,z,w = unitquat.narrow(1,0,1), unitquat.narrow(1,1,1), unitquat.narrow(1,2,1), unitquat.narrow(1,3,1)
		dRdqh_w = 2 * torch.cat([ w, -z,  y,  z,  w, -x, -y,  x,  w]).view(N, 9, 1) # Eqn 33, rows first
		dRdqh_x = 2 * torch.cat([ x,  y,  z,  y, -x, -w,  z,  w, -x]).view(N, 9, 1) # Eqn 34, rows first
		dRdqh_y = 2 * torch.cat([-y,  x,  w,  x,  y,  z, -w,  z, -y]).view(N, 9, 1) # Eqn 35, rows first
		dRdqh_z = 2 * torch.cat([-z, -w,  x,  w, -z,  y,  x,  y,  z]).view(N, 9, 1) # Eqn 36, rows first
		dRdqh   = torch.cat([dRdqh_x, dRdqh_y, dRdqh_z, dRdqh_w], 3) 				# N x 9 x 4
		return dRdqh

	# Compute the derivatives of a unit quaternion w.r.t a quaternion
	def compute_grad_unitquat_wrt_quat(self, unitquat, quat):
		# Compute the quaternion norms
		N = quat.size(0)
		norm2 = (quat * quat).sum(1) # Norm-squared
		norm  = torch.sqrt(norm2)    # Length of the quaternion

		# Compute gradient dq'/dq
		# TODO: No check for normalization issues currently
		I  	  = torch.eye(4).view(1,4,4).expand(N,4,4).type_as(quat)
		qQ 	  = torch.bmm(unitquat, unitquat.transpose(1,2)) # q'*q'^T
		dqhdq = (I - qQ) / (norm.view(N,1,1).expand_as(I))

		# Return
		return dqhdq

	########
	#### Stereographic Projection Quaternion representation helpers
	# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 42-45)
	# The singularity (origin) has been moved from [0,0,0,-1] to [-1,0,0,0], so the 0 rotation quaternion [1,0,0,0] maps to [0,0,0] as opposed of to [1,0,0].
	# For all equations, just assume that the quaternion is (qx,qy,qz,qw) instead of the pdf convention -> (qw,qx,qy,qz)
	# Similar to: https://github.com/FugroRoames/Rotations.jl

	# Compute Unit Quaternion from SP-Quaternion
	def create_unitquat_from_spquat(self, spquat):
		# Init memory
		N = spquat.size(0)
		unitquat = spquat.new().resize(N,4).fill_(0);

		# Compute the unit quaternion (qx, qy, qz, qw)
		x,y,z  = spquat[:,0], spquat[:,1], spquat[:,2]
		alpha2 = x*x + y*y + z*z  				 # x^2 + y^2 + z^2
		unitquat[:, 0] = (2*x) / (1+alpha2)		 # qx
		unitquat[:, 0] = (2*y) / (1+alpha2)		 # qy
		unitquat[:, 0] = (2*z) / (1+alpha2)		 # qz
		unitquat[:, 3] = (1-alpha2) / (1+alpha2) # qw
		return unitquat

	# Compute the derivatives of a unit quaternion w.r.t a SP quaternion
	# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 42-45)
	def compute_grad_unitquat_wrt_spquat(self, spquat):
		# Compute scalars
		N = spquat.size(0)
		x,y,z  = spquat.narrow(1,0,1), spquat.narrow(1,1,1), spquat.narrow(1,2,1)
		x2,y2,z2 = x*x, y*y, z*z
		s  = 1 + x2 + y2 + z2   # 1 + x^2 + y^2 + z^2 = 1 + alpha^2
		s2 = (s*s).expand(N, 4) # (1 + alpha^2)^2

		# Compute gradient dq'/dspq
		dqhdspq_x = (torch.cat([ 2*s-4*x2, -4*x*y, -4*x*z, -4*x ]) / s2).view(N,4,1)
		dqhdspq_y = (torch.cat([-4*x*y,  2*s-4*y2, -4*y*z, -4*y ]) / s2).view(N,4,1)

	def forward(self, input):
		# Check dimensions
		batch_size, num_se3, num_rows, num_cols = input.size()
		assert (num_rows == 3 and num_cols == 5);

		# Init for FWD pass
		self.save_for_backward(input)
		input_v = input.view(-1, 3, 5);
		r = input_v.narrow(2, 0, 3);
		t = input_v.narrow(2, 3, 1);
		p = input_v.narrow(2, 4, 1);

		# Compute output = [r, t + p - Rp]
		output = input.new().resize_(batch_size, num_se3, 3, 4);
		output.narrow(3, 0, 3).copy_(r); # r
		output.narrow(3, 3, 1).copy_(t+p).add_(-1, torch.bmm(r, p)); # t + p - Rp

		# Return
		return output;

	def backward(self, grad_output):
		# Get saved tensors & setup vars
		input = self.saved_tensors[0]
		input_v = input.view(-1, 3, 5);
		r = input_v.narrow(2, 0, 3);
		p = input_v.narrow(2, 4, 1);
		ro_g = grad_output.view(-1, 3, 4).narrow(2, 0, 3);
		to_g = grad_output.view(-1, 3, 4).narrow(2, 3, 1);

		# Initialize grad input
		input_g = input.new().resize_as_(input);
		input_g.narrow(3,0,3).copy_(ro_g).add_(-1, torch.bmm(to_g, p.transpose(1,2))); # r_g = ro_g - (to_g * p^T)
		input_g.narrow(3,3,1).copy_(to_g); 											  # t_g = to_g
		input_g.narrow(3,4,1).copy_(to_g).add_(-1, torch.bmm(r.transpose(1,2), to_g)); # p_g = to_g - (R^T * to_g)

		# Return
		return input_g;


## FWD/BWD pass module
class SE3ToRt(Module):
	def __init__(self, transform_type='affine', has_pivot=False):
		super(SE3ToRt, self).__init__()
		self.transform_type = transform_type
		self.has_pivot		= has_pivot
		self.eps			= 1e-12

	def forward(self, input):
		return SE3ToRtFunction()(input, self.transform_type, self.has_pivot)