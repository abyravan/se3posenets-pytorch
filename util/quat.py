import torch
import numpy as np
import pyquaternion as pyq

## Quaternion to rotation matrix
def quat_xyzw_to_rot(_quat):
    quat = _quat.view(4)
    q = pyq.Quaternion(array=np.array([quat[3], quat[0], quat[1], quat[2]], np.float64))
    return torch.from_numpy(q.rotation_matrix).type_as(_quat)

## Quaternion from rotation matrix (3x3 or 4x4)
def rot_to_quat_xyzw(rot):
    """Initialise from matrix representation
    Create a Quaternion by specifying the 3x3 rotation or 4x4 transformation matrix
    (as a numpy array) from which the quaternion's rotation should be created.
    From: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
    """
    # Return
    quat = pyq.Quaternion(matrix=rot.cpu().clone().numpy()) # wxyz
    return torch.Tensor([quat[1], quat[2], quat[3], quat[0]]).type_as(rot) # xyzw

## Interpolate "N" quaternions between start & goal quaternions
## Returns "N" quaternions
def interpolate_quat_xyzw(start, goal, N, include_endpoints=True):
    st, gl = start.view(4), goal.view(4)
    stq = pyq.Quaternion(array=np.array([st[3], st[0], st[1], st[2]], np.float64)) # Convert to wxyz from xyzw
    glq = pyq.Quaternion(array=np.array([gl[3], gl[0], gl[1], gl[2]], np.float64))
    outq = [] # Output
    for q in pyq.Quaternion.intermediates(stq, glq, N, include_endpoints):
        outq.append([q[1], q[2], q[3], q[0]]) # convert to xyzw from wxyz
    return torch.Tensor(outq).type_as(start)

## Interpolate between two rotation matrices (internally convert to quats)
def interpolate_rot_matrices(start, goal, N, include_endpoints=True):
    stq = pyq.Quaternion(matrix=start.cpu().clone().numpy()) # wxyz
    glq = pyq.Quaternion(matrix=goal.cpu().clone().numpy())  # wxyz
    outrot = []  # Output
    for q in pyq.Quaternion.intermediates(stq, glq, N, include_endpoints):
        outrot.append(torch.from_numpy(q.rotation_matrix[np.newaxis,:,:]))
    return torch.cat(outrot, 0).type_as(start) # N x 3 x 3

#################### TEST
''' 
import utilquat as uq
import torch
q1, q2 = torch.Tensor([0,0,0,1]), torch.Tensor([1,0,0,0])
r1, r2 = uq.quat_xyzw_to_rot(q1), uq.quat_xyzw_to_rot(q2)

qinter = uq.interpolate_quat_xyzw(q1, q2, 9, True)
rinter_1 = []
for k in xrange(qinter.size(0)):
    rinter_1.append(uq.quat_xyzw_to_rot(qinter[k]).unsqueeze(0))
rinter_1 = torch.cat(rinter_1, 0)
rinter = uq.interpolate_rot_matrices(r1, r2, 9, True)
'''