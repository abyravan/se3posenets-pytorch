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