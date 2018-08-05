import h5py
import matplotlib.pyplot as plt

import numpy as np
import io
from PIL import Image
import sys

import torch
import cv2

opengl = False

def RGBToDepth(img):
    return img[:,:,0]+.01*img[:,:,1]+.0001*img[:,:,2]

def RGBAToMask(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        #print(i, dim, shift, buf[0,0,dim], np.left_shift(buf[0,0,dim], shift))
        mask += np.left_shift(buf[:,:, dim], shift)
    return mask

def RGBAArrayToMasks(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        mask += np.left_shift(buf[:,:,:, dim], shift)
    return mask

def PNGToNumpy(png):
    stream = io.BytesIO(png)
    im = Image.open(stream)
    return np.asarray(im, dtype=np.uint8)

def ConvertPNGListToNumpy(data):
    imgs = []
    for raw in data:
        imgs.append(PNGToNumpy(raw))
    arr = np.array(imgs)
    return arr

def ConvertDepthPNGListToNumpy(data):
    imgs = []
    for raw in data:
        imgs.append(RGBToDepth(PNGToNumpy(raw)))
    arr = np.array(imgs)
    return arr

########
# Project a 3D point to an image using the pinhole camera model (perspective transform)
# Given a camera matrix of the form [fx 0 cx; 0 fy cy; 0 0 1] (x = cols, y = rows) and a 3D point (x,y,z)
# We do: x' = x/z, y' = y/z, [px; py] = cameraMatrix * [x'; y'; 1]
# Returns a 2D pixel (px, py)
def project_to_image(camera_intrinsics, point):
    # Project to (0,0,0) if z = 0
    pointv = point.view(4) # 3D point
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
    pt =  torch.mm(pose.view(4,4), point.view(4,1))
    return pt
    #posev, pointv = pose.view(3,4), point.view(3,1)
    #return torch.mm(posev[:,0:3], pointv).view(3) + posev[:,3] # R*p + t

# Plot a 3d frame (X,Y,Z axes) of an object on a qt window
# given the 6d pose of the object (3x4 matrix) in the camera frame of reference,
# and the camera's projection matrix (3x3 matrix of form [fx 0 cx; 0 fy cy; 0 0 1])
# Img represented as H x W x 3 (numpy array) & Pose is a 3 x 4 torch tensor
def draw_3d_frame(img, pose, camera_intrinsics={}, pixlength=10.0, thickness=2):
    # Project the principal vectors (3 columns which denote the {X,Y,Z} vectors of the object) into the global (camera frame)
    dv = 0.2 # Length of 3D vector
    X = transform(pose, torch.FloatTensor([dv, 0, 0, 1]))
    Y = transform(pose, torch.FloatTensor([ 0,dv, 0, 1]))
    Z = transform(pose, torch.FloatTensor([ 0, 0,dv, 1]))
    O = transform(pose, torch.FloatTensor([ 0, 0, 0, 1]))
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
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Xp.numpy()), [1,0,0], thickness)
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Yp.numpy()), [0,1,0], thickness)
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Zp.numpy()), [0,0,1], thickness)

##########
def opengl_project(point, width, height):
    pointv = point.view(4) # 3D point
    x = (pointv[0] + 1) * width * 0.5
    y = (1 - pointv[1]) * height * 0.5
    return torch.Tensor([x,y]).type_as(point)

# Transform a point through the given pose (point in pose's frame of reference to global frame of reference)
# Pose: (3x4 matrix) [R | t]
# Point: (position) == torch.Tensor(3)
# Returns: R*p + t == torch.Tensor(3)
def opengl_transform(pose, point):
    pt =  torch.mm(pose.view(4,4), point.view(4,1))
    pt /= pt[3]
    return pt
    #posev, pointv = pose.view(3,4), point.view(3,1)
    #return torch.mm(posev[:,0:3], pointv).view(3) + posev[:,3] # R*p + t

# Plot a 3d frame (X,Y,Z axes) of an object on a qt window
# given the 6d pose of the object (3x4 matrix) in the camera frame of reference,
# and the camera's projection matrix (3x3 matrix of form [fx 0 cx; 0 fy cy; 0 0 1])
# Img represented as H x W x 3 (numpy array) & Pose is a 3 x 4 torch tensor
def opengl_draw_3d_frame(img, pose, pixlength=10.0, thickness=2):
    # Project the principal vectors (3 columns which denote the {X,Y,Z} vectors of the object) into the global (camera frame)
    dv = 0.2 # Length of 3D vector
    X = opengl_transform(pose, torch.FloatTensor([dv, 0, 0, 1]))
    Y = opengl_transform(pose, torch.FloatTensor([ 0,dv, 0, 1]))
    Z = opengl_transform(pose, torch.FloatTensor([ 0, 0,dv, 1]))
    O = opengl_transform(pose, torch.FloatTensor([ 0, 0, 0, 1]))
    # Project the end-points of the vectors and the frame origin to the image to get the corresponding pixels
    Xp = opengl_project(X, img.shape[1], img.shape[0])
    Yp = opengl_project(Y, img.shape[1], img.shape[0])
    Zp = opengl_project(Z, img.shape[1], img.shape[0])
    Op = opengl_project(O, img.shape[1], img.shape[0])
    # Maintain a specific length in pixel space by changing the tips of the frames to match correspondingly
    unitdirX = (Xp-Op).div_((Xp-Op).norm(2) + 1e-12) # Normalize it
    unitdirY = (Yp-Op).div_((Yp-Op).norm(2) + 1e-12) # Normalize it
    unitdirZ = (Zp-Op).div_((Zp-Op).norm(2) + 1e-12) # Normalize it
    Xp = Op + pixlength * unitdirX
    Yp = Op + pixlength * unitdirY
    Zp = Op + pixlength * unitdirZ
    # Draw lines on the image
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Xp.numpy()), [1,0,0], thickness)
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Yp.numpy()), [0,1,0], thickness)
    cv2.line(img.numpy(), tuple(Op.numpy()), tuple(Zp.numpy()), [0,0,1], thickness)

########
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
    rt = torch.zeros(_pose.size(0), _pose.size(1), 4, 4).type_as(_pose)
    rt[:,:,0:3,0:3] = quat_to_rot(quat.clone()).view(_pose.size(0), _pose.size(1), 3, 3)
    rt[:,:,0:3,3]   = pos.contiguous().view(_pose.size(0), _pose.size(1), 3)
    rt[:,:,3,3]     = 1.0
    return rt

# Rotation about the Y-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
def create_roty(theta):
    N = theta.size(0)
    thetas = theta.squeeze()
    rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(thetas)  # (DO NOT use expand as it does not allocate new memory)
    rot[:, 0, 0] = torch.cos(thetas)
    rot[:, 2, 2] = rot[:, 0, 0]
    rot[:, 2, 0] = torch.sin(thetas)
    rot[:, 0, 2] = -rot[:, 2, 0]
    return rot

# Rotation about the Z-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
def create_rotz(theta):
    N = theta.size(0)
    thetas = theta.squeeze()
    rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(theta)  # (DO NOT use expand as it does not allocate new memory)
    rot[:, 0, 0] = torch.cos(thetas)
    rot[:, 1, 1] = rot[:, 0, 0]
    rot[:, 0, 1] = torch.sin(thetas)
    rot[:, 1, 0] = -rot[:, 0, 1]
    return rot

# def get_3d_frame(pose):
#     # Project the principal vectors (3 columns which denote the {X,Y,Z} vectors of the object) into the global (camera frame)
#     dv = 0.2  # Length of 3D vector
#     X = transform(pose, torch.FloatTensor([dv, 0, 0]))
#     Y = transform(pose, torch.FloatTensor([0, dv, 0]))
#     Z = transform(pose, torch.FloatTensor([0, 0, dv]))
#     O = transform(pose, torch.FloatTensor([0, 0, 0]))
#     return X,Y,Z,O

#data = h5py.File("blocks_babble01336_2018-07-18_09-45-01_arm=right_obj=blue00.failure.h5")
#data = h5py.File("blocks_babble01333_2018-07-18_09-43-20_arm=right_obj=yellow00.success.h5")
import os
files = os.listdir(sys.argv[1])
np.random.seed(1500)
perm = np.random.permutation(len(files))
for k in range(len(perm)):
    filename = os.path.join(sys.argv[1], files[perm[k]])
    if (filename.find(".h5") != -1) and (filename.find("success") != -1):
        print("Loading data from file: {}".format(filename))
        break
data = h5py.File(filename) #h5py.File(sys.argv[1])

num_imgs = len(data['images_rgb'])
max_rgb, max_depth = 255., 2.
rgbs   = ConvertPNGListToNumpy(data['images_rgb']) / max_rgb
depths = ConvertDepthPNGListToNumpy(data['images_depth'])
masks  = RGBAArrayToMasks(ConvertPNGListToNumpy(data['images_mask']))
objids  = (masks & ((1<<24)-1)) * (masks >= 0)
linkids = ((masks >> 24)-1) * (masks >= 0)

# Find unique object IDs
vals = []
poses = []
for j in data.keys():
    if (j.find("pose") != -1):
        poses.append(np.array(data[j]))
        vals.append(int(j[4:]))

# Make sure that the BG poses are initialized to identity
sz = None
for k in range(len(poses)):
    if poses[k].size != 0:
        sz = poses[k].shape
        break
for k in range(len(poses)):
    if poses[k].size == 0:
        poses[k] = np.zeros(sz)
        poses[k][:,-1] = 1 # Unit quaternion for identity rotation
poses = np.array(poses)

# Convert quaternions to rotation matrices (basically get 3x4 matrices)
rtposes = se3quat_to_rt(torch.from_numpy(poses))
rtposes_inv = torch.cat([r.inverse() for r in rtposes.view(-1,4,4).clone()]).view_as(rtposes).clone()

# Figure out camera params
img_width = rgbs[0].shape[1] # width in pixels
img_height = rgbs[0].shape[0] # width in pixels
vfov = (np.pi/180.) * np.array(data['camera_fov'])
fx = fy = img_height/(2*np.tan(vfov/2))
cx, cy = 0.5*img_width, 0.5*img_height
camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': img_width}
camera_pose = np.array(data['camera_view_matrix']).reshape(4,4).transpose() # I think this takes a point in the global frame of reference and transforms it to camera frame

roty = create_roty(torch.Tensor([np.pi])).squeeze().numpy()
rotyT = np.eye(4,4)
rotyT[:3,:3] = roty
camera_pose = rotyT.dot(camera_pose)

rotz = create_rotz(torch.Tensor([np.pi])).squeeze().numpy()
rotzT = np.eye(4,4)
rotzT[:3,:3] = rotz
camera_pose = rotzT.dot(camera_pose)

camera_pose = torch.from_numpy(camera_pose).clone()
camera_pose_inv = camera_pose.inverse().clone()

# Transform all poses from global frame of reference to camera frame
N = rtposes.size(0) * rtposes.size(1)
rtposes_cam = torch.bmm(camera_pose.view(1,4,4).expand(N,4,4).clone(),
                        rtposes.view(-1, 4, 4).clone())
# rtposes_cam = torch.bmm(camera_pose_inv.view(1,4,4).expand(N,4,4).clone(),
#                         rtposes.view(-1, 4, 4).clone())
# rtposes_cam = torch.bmm(camera_pose.view(1,4,4).expand(N,4,4).clone(),
#                         rtposes_inv.view(-1, 4, 4).clone())
# rtposes_cam = torch.bmm(camera_pose_inv.view(1,4,4).expand(N,4,4).clone(),
#                         rtposes_inv.view(-1, 4, 4).clone())
# rtposes_cam = torch.bmm(rtposes.view(-1, 4, 4).clone(),
#                         camera_pose.view(1,4,4).expand(N,4,4).clone())
# rtposes_cam = torch.bmm(rtposes_inv.view(-1, 4, 4).clone(),
#                         camera_pose.view(1,4,4).expand(N,4,4).clone())
# rtposes_cam = torch.bmm(rtposes.view(-1, 4, 4).clone(),
#                         camera_pose_inv.view(1,4,4).expand(N,4,4).clone())
# rtposes_cam = torch.bmm(rtposes_inv.view(-1, 4, 4).clone(),
#                         camera_pose_inv.view(1,4,4).expand(N,4,4).clone())

# Camera projection matrix
proj_matrix = torch.from_numpy(np.array(data['camera_projection_matrix']).reshape(4,4)).transpose(0,1).clone()
proj_matrix_t = proj_matrix.transpose(0,1).clone()

if opengl:
    rtposes_cam_1 = torch.bmm(proj_matrix.view(1,4,4).expand(N,4,4).clone(),
                              rtposes_cam.view(-1, 4, 4).clone())
else:
    rtposes_cam_1 = rtposes_cam
# rtposes_cam_1 = torch.bmm(proj_matrix_t.view(1,4,4).expand(N,4,4).clone(),
#                           rtposes_cam.view(-1, 4, 4).clone())

# Convert to float
rtposes_f = rtposes.float()
rtposes_cam_f = rtposes_cam_1.view_as(rtposes_f).float()

# remap mask values
masksc = np.copy(masks)
d    = {vals[j]:j for j in range(len(vals))}
for ky, vl in d.items(): masksc[masks == ky] = vl

# Display the data
plt.ion()
fig = plt.figure(100)

# # 3D plot
# from mpl_toolkits.mplot3d import Axes3D
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# fig1.hold(True)

# Render
try:
    os.makedirs(sys.argv[1] + "/test/")
except:
    pass
for k in range(0,len(rgbs),4):

    # # Plot 3D poses
    # plt.figure(101)
    # ax.clear()
    # for j in xrange(len(poses)):
    #     if j not in [15, 28, 37, 38]:
    #         continue
    #     x,y,z,o = get_3d_frame(rtposes_f[j][k][:-1,:])
    #     ax.quiver(o[0], o[1], o[2], x[0], x[1], x[2])
    #     ax.quiver(o[0], o[1], o[2], y[0], y[1], y[2])
    #     ax.quiver(o[0], o[1], o[2], z[0], z[1], z[2])

    # Project poses onto RGB images
    rgbout = np.concatenate([rgbs[k][:,:,2:], rgbs[k][:,:,1:2], rgbs[k][:,:,0:1]], -1)
    cv2.imwrite(sys.argv[1] + "/test/rgb{}.png".format(k), rgbout * 255.0)
    for j in range(len(poses)):
        if opengl:
            opengl_draw_3d_frame(torch.from_numpy(rgbs[k]), rtposes_cam_f[j][k])
        else:
            draw_3d_frame(torch.from_numpy(rgbs[k]), rtposes_cam_f[j][k],
                          camera_intrinsics=camera_intrinsics)

    # Show rgb, depth, masks
    plt.figure(100)
    fig.suptitle("Image: {}/{}".format(k, len(rgbs)))
    plt.subplot(131); plt.imshow(rgbs[k]); plt.title("RGB")
    plt.subplot(132); plt.imshow(depths[k]); plt.title("Depth")
    plt.subplot(133); plt.imshow(masksc[k]); plt.title("Segmentation Mask")
    #cv2.imwrite("/Users/barun/Desktop/Research/Projects/gazebo-learning-planning/data/masks{}.png".format(k), masksc[k])

    rgbout = np.concatenate([rgbs[k][:,:,2:], rgbs[k][:,:,1:2], rgbs[k][:,:,0:1]], -1)
    cv2.imwrite(sys.argv[1] + "/test/rgbp{}.png".format(k), rgbout * 255.0)

    # Show rgb, depth, obj ids, link ids
    # plt.subplot(141)
    # plt.imshow(rgbs[k])
    # plt.subplot(142)
    # plt.imshow(depths[k])
    # plt.subplot(143)
    # plt.imshow(objids[k])
    # plt.subplot(144)
    # plt.imshow(linkids[k])

    plt.draw()
    plt.pause(0.02)
    if k % 5 == 0:
        plt.clf()

## todo: Get poses for masks, project them onto the frame using cam params
## todo: parse joint data, parse camera params and modelview etc
