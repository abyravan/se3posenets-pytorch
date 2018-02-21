import torch

#### General helpers
# Create a skew-symmetric matrix "S" of size [B x 3 x 3] (passed in) given a [B x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    output = torch.zeros(N,3,3).type_as(vector)
    output[:, 0, 1] = -vector[:, 2]
    output[:, 1, 0] =  vector[:, 2]
    output[:, 0, 2] =  vector[:, 1]
    output[:, 2, 0] = -vector[:, 1]
    output[:, 1, 2] = -vector[:, 0]
    output[:, 2, 1] =  vector[:, 0]
    return output

# Compute the rotation matrix R from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def create_rot_from_aa(params):
    # Get the un-normalized axis and angle
    N, eps = params.size(0), 1e-12
    axis = params.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle
    small = (angle2 < eps)

    # Create Identity matrix
    I = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(params)

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute A = (sin(theta)/theta)
    A = torch.sin(angle) / angle
    A[small] = 1.0 # sin(0)/0 ~= 1

    # Compute B = (1 - cos(theta)/theta^2)
    B = (1 - torch.cos(angle)) / angle2
    B[small] = 1/2 # lim 0-> 0 (1 - cos(0))/0^2 = 1/2

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    R = I + K * A.expand(N, 3, 3) + K2 * B.expand(N, 3, 3)
    return R

# Compute the rotation matrix R & translation vector from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def AAToRt(params):
    # Check dimensions
    bsz, nse3, ndim = params.size()
    N = bsz*nse3
    assert (ndim == 6)

    # Trans | Rot params
    params_v = params.view(N, ndim, 1).clone()
    rotparam   = params_v.narrow(1,3,3) # so(3)
    transparam = params_v.narrow(1,0,3) # R^3

    # Compute rotation matrix (Bk x 3 x 3)
    R = create_rot_from_aa(rotparam)

    # Final tfm
    return torch.cat([R, transparam], 2).view(bsz,nse3,3,4).clone() # B x K x 3 x 4

# Compute the jacobians of the 3x4 transform matrix w.r.t transform parameters
def AAToRtJac(params):
    # Check dimensions
    bsz, nse3, ndim = params.size()
    N = bsz * nse3
    eps = 1e-12
    assert (ndim == 6)

    # Create jacobian matrix
    J = torch.zeros(bsz*bsz, nse3*nse3, 3, 4, ndim).type_as(params)
    J[::(bsz+1), ::(nse3+1), 0, 3, 0] = 1 # Translation vector passed out as is (first 3 params are translations)
    J[::(bsz+1), ::(nse3+1), 1, 3, 1] = 1 # Translation vector passed out as is (first 3 params are translations)
    J[::(bsz+1), ::(nse3+1), 2, 3, 2] = 1 # Translation vector passed out as is (first 3 params are translations)

    # Trans | Rot params
    params_v = params.view(N, ndim, 1).clone()
    rotparam = params_v.narrow(1, 3, 3)  # so(3)

    # Get the un-normalized axis and angle
    axis = rotparam.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    small = (angle2 < eps)  # Don't need gradient w.r.t this operation (also because of pytorch error: https://discuss.pytorch.org/t/get-error-message-maskedfill-cant-differentiate-the-mask/9129/4)

    # Compute rotation matrix
    R = create_rot_from_aa(rotparam)

    # Create Identity matrix
    I = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(params)

    ######## Jacobian computation
    # Compute: v x (Id - R) for all the columns of (Id-R)
    vI = torch.cross(axis.expand_as(I), (I - R), 1)  # (Bk) x 3 x 3 => v x (Id - R)

    # Compute [v * v' + v x (Id - R)] / ||v||^2
    vV = torch.bmm(axis, axis.transpose(1, 2))  # (Bk) x 3 x 3 => v * v'
    vV = (vV + vI) / (angle2.view(N, 1, 1).expand_as(vV))  # (Bk) x 3 x 3 => [v * v' + v x (Id - R)] / ||v||^2

    # Iterate over the 3-axis angle parameters to compute their gradients
    # ([v * v' + v x (Id - R)] / ||v||^2 _ k) x (R) .* gradOutput  where "x" is the cross product
    for k in range(3):
        # Create skew symmetric matrix
        skewsym = create_skew_symmetric_matrix(vV.narrow(2, k, 1))

        # For those AAs with angle^2 < threshold, gradient is different
        # We assume angle = 0 for these AAs and update the skew-symmetric matrix to be one w.r.t identity
        if (small.any()):
            vec = torch.zeros(1, 3).type_as(skewsym)
            vec[0,k] = 1  # Unit vector
            idskewsym = create_skew_symmetric_matrix(vec)
            for i in range(N):
                if (angle2[i].squeeze()[0] < eps):
                    skewsym[i].copy_(idskewsym)  # Use the new skew sym matrix (around identity)

        # Compute the jacobian elements now
        J[::(bsz+1), ::(nse3+1), :, 0:3, k+3] = torch.bmm(skewsym, R) # (Bk) x 3 x 3

    return J.view(bsz, bsz, nse3, nse3, 12, ndim).permute(0,2,4,1,3,5).clone().view(bsz*nse3*12, bsz*nse3*ndim).clone()


# ###########
## Setup stuff
bsz, nse3 = 2, 3
tensortype = 'torch.DoubleTensor'
params = torch.rand(bsz, nse3, 6).type(tensortype) # AA params
initparams = params.clone()

##########
### Finite difference to check stuff
tfmb  = AAToRt(params).clone()
jacb  = AAToRtJac(params)
jacf  = jacb.clone().zero_()
params_v = params.view(-1)
eps = 1e-6
for k in range(len(params_v)):
    params_v[k] += eps # Perturb
    tfmf = AAToRt(params.clone()).clone()
    jacf[:,k] = (tfmf - tfmb) / eps
    params_v[k] -= eps # Reset
diff = jacf - jacb
print('Diff: ', diff.max(), diff.min())

# ### Finite difference to check stuff
# import se3layers as se3nn
# tfmb_1  = se3nn.SE3ToRt('se3aa')(torch.autograd.Variable(params))
# jacb_1  = AAToRtJac(params)
# jacf_1  = jacb_1.clone().zero_()
# params_v_1 = params.view(-1)
# eps = 1e-6
# for k in range(len(params_v_1)):
#     params_v_1[k] += eps # Perturb
#     tfmf_1 = se3nn.SE3ToRt('se3aa')(torch.autograd.Variable(params))
#     jacf_1[:,k] = (tfmf_1 - tfmb_1).data / eps
#     params_v_1[k] -= eps # Reset
# diff_1 = jacf_1 - jacb_1
# print('Diff [SE3TORT]: ', diff_1.max(), diff_1.min())
