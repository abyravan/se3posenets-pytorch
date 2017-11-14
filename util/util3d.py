import torch
import cv2

# Project a 3D point to an image using the pinhole camera model (perspective transform)
# Given a camera matrix of the form [fx 0 cx; 0 fy cy; 0 0 1] (x = cols, y = rows) and a 3D point (x,y,z)
# We do: x' = x/z, y' = y/z, [px; py] = cameraMatrix * [x'; y'; 1]
# Returns a 2D pixel (px, py)
def project_to_image(camera_intrinsics, point):
    # Project to (0,0,0) if z = 0
    pointv = point.view(3) # 3D point
    if pointv[2] == 0:
        return torch.zeros(2).type_as(point)

    # Perspective projection
    c = camera_intrinsics['fx'] * (pointv[0] / pointv[2]) + camera_intrinsics['cx'] # fx * (x/z) + cx
    r = camera_intrinsics['fy'] * (pointv[1] / pointv[2]) + camera_intrinsics['cy'] # fy * (y/z) + cy
    return torch.Tensor([c,r]).type_as(point)

# Transform a point through the given pose (point in pose's frame of reference to global frame of reference)
# Pose: (3x4 matrix) [R | t]
# Point: (position) == torch.Tensor(3)
# Returns: R*p + t == torch.Tensor(3)
def transform(pose, point):
    posev, pointv = pose.view(3,4), point.view(3,1)
    return torch.mm(posev[:,0:3], pointv).view(3) + posev[:,3] # R*p + t

# Plot a 3d frame (X,Y,Z axes) of an object on a qt window
# given the 6d pose of the object (3x4 matrix) in the camera frame of reference,
# and the camera's projection matrix (3x3 matrix of form [fx 0 cx; 0 fy cy; 0 0 1])
# Img represented as H x W x 3 (numpy array) & Pose is a 3 x 4 torch tensor
def draw_3d_frame(img, pose, color=[], camera_intrinsics={}, pixlength=10.0, thickness=2):
    # Project the principal vectors (3 columns which denote the {X,Y,Z} vectors of the object) into the global (camera frame)
    dv = 0.2 # Length of 3D vector
    X = transform(pose, torch.FloatTensor([dv, 0, 0]))
    Y = transform(pose, torch.FloatTensor([ 0,dv, 0]))
    Z = transform(pose, torch.FloatTensor([ 0, 0,dv]))
    O = transform(pose, torch.FloatTensor([ 0, 0, 0]))
    # Project the end-points of the vectors and the frame origin to the image to get the corresponding pixels
    Xp = project_to_image(camera_intrinsics, X)
    Yp = project_to_image(camera_intrinsics, Y)
    Zp = project_to_image(camera_intrinsics, Z)
    Op = project_to_image(camera_intrinsics, O)
    # Maintain a specific length in pixel space by changing the tips of the frames to match correspondingly
    unitdirX = (Xp-Op).div_((Xp-Op).norm(2) + 1e-12) # Normalize it
    unitdirY = (Yp-Op).div_((Yp-Op).norm(2) + 1e-12) # Normalize it
    unitdirZ = (Zp-Op).div_((Zp-Op).norm(2) + 1e-12) # Normalize it
    Xp = Op + pixlength * unitdirX
    Yp = Op + pixlength * unitdirY
    Zp = Op + pixlength * unitdirZ
    # Draw lines on the image
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Xp.numpy()), color, thickness)
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Yp.numpy()), color, thickness)
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Zp.numpy()), color, thickness)

# Compute the rotation matrix R from a set of unit-quaternions (N x 4):
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 9)
def create_rot_from_unitquat(unitquat):
    # Init memory
    N = unitquat.size(0)
    rot = unitquat.new().resize_(N, 3, 3)

    # Get quaternion elements. Quat = [qx,qy,qz,qw] with the scalar at the rear
    x, y, z, w = unitquat[:, 0], unitquat[:, 1], unitquat[:, 2], unitquat[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w

    # Row 1
    rot[:, 0, 0] = w2 + x2 - y2 - z2  # rot(0,0) = w^2 + x^2 - y^2 - z^2
    rot[:, 0, 1] = 2 * (x * y - w * z)  # rot(0,1) = 2*x*y - 2*w*z
    rot[:, 0, 2] = 2 * (x * z + w * y)  # rot(0,2) = 2*x*z + 2*w*y

    # Row 2
    rot[:, 1, 0] = 2 * (x * y + w * z)  # rot(1,0) = 2*x*y + 2*w*z
    rot[:, 1, 1] = w2 - x2 + y2 - z2  # rot(1,1) = w^2 - x^2 + y^2 - z^2
    rot[:, 1, 2] = 2 * (y * z - w * x)  # rot(1,2) = 2*y*z - 2*w*x

    # Row 3
    rot[:, 2, 0] = 2 * (x * z - w * y)  # rot(2,0) = 2*x*z - 2*w*y
    rot[:, 2, 1] = 2 * (y * z + w * x)  # rot(2,1) = 2*y*z + 2*w*x
    rot[:, 2, 2] = w2 - x2 - y2 + z2  # rot(2,2) = w^2 - x^2 - y^2 + z^2

    # Return
    return rot

## Quaternion to rotation matrix
def quat_to_rot(_quat):
    # Compute the unit quaternion
    quat = _quat.view(-1, 4).clone() # Get the quaternions
    unitquat = torch.nn.functional.normalize(quat, p=2, dim=1, eps=1e-12)  # self.create_unitquat_from_quat(rot_params)

    # Compute rotation matrix from unit quaternion
    return create_rot_from_unitquat(unitquat)

## SE3-Quat to Rt
def se3quat_to_rt(_pose):
    pose = _pose.view(-1, 7).clone() # Get poses
    pos, quat = pose[:,0:3], pose[:,3:] # Position, Quaternion
    rt = torch.zeros(pose.size(0), 4, 4)
    rt[:,0:3,0:3] = quat_to_rot(quat)
    rt[:,0:3,3]   = pos
    rt[:,3,3] = 1.0 # Last row is 0,0,0,1
    return rt