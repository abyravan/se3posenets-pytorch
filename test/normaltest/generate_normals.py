import data
import torch
import numpy as np
import se3layers as se3nn
from layers._ext import se3layers
import util
import cv2

## Real vs sim data
realdata = True
ht, wd = 240, 320
if realdata:
    datadir = '/home/barun/Projects/se3nets-pytorch/test/normaltest/realdata/'
    cam_extrinsics = data.read_cameradata_file(datadir + '/cameradata.txt')
    # Setup camera intrinsics
    cam_intrinsics = {'fx': 525.0 / 2,
                      'fy': 525.0 / 2,
                      'cx': 319.5 / 2,
                      'cy': 239.5 / 2}
    suffix, scale = '', 1e-3
else:
    #datadir = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/learn_physics_models/data/singlebox_fixsz_visbullet_2016_04_20_20_55_18_1/bag1002/'
    datadir = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/olddata/baxter_babbling_rarm_3min/postprocessmotions/motion0/'
    cam_extrinsics = data.read_cameradata_file(datadir + '/../cameradata.txt')

    # Setup camera intrinsics
    cam_intrinsics = {'fx': 589.3664541825391 / 2,
                      'fy': 589.3664541825391 / 2,
                      'cx': 320.5 / 2,
                      'cy': 240.5 / 2}
    suffix, scale = 'sub', 1e-4

print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(ht, wd,
                                                                            cam_intrinsics['fx'],
                                                                            cam_intrinsics['fy'],
                                                                            cam_intrinsics['cx'],
                                                                            cam_intrinsics['cy']))
cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(ht, wd,
                                                                      cam_intrinsics)

# Read depth, labels & clouds (t = 0)
depth_1  = data.read_depth_image(datadir+'/depth'+suffix+'1000.png',ht,wd,scale).view(1,1,ht,wd).clone()
labels_1 = data.read_label_image(datadir+'/labels'+suffix+'1000.png',ht,wd).view(1,1,ht,wd).clone()
se3state_1 = data.read_baxter_se3state_file(datadir+'/se3state1000.txt')
poses_1    = torch.Tensor(1, len(se3state_1)+1, 3, 4).fill_(0) # Setup size
poses_1[0,0,:,0:3] = torch.eye(3).float()  # Identity transform for BG
for id, tfm in se3state_1.items():
    se3tfm = torch.mm(cam_extrinsics['modelView'], tfm)  # NOTE: Do matrix multiply, not * (cmul) here. Camera data is part of options
    poses_1[0][id] = se3tfm[0:3, :] # 3 x 4 transform (id is 1-indexed already, 0 is BG)

# Read depth, labels & clouds (t = 10)
depth_2  = data.read_depth_image(datadir+'/depth'+suffix+'2000.png',ht,wd,scale).view(1,1,ht,wd).clone()
labels_2 = data.read_label_image(datadir+'/labels'+suffix+'2000.png',ht,wd).view(1,1,ht,wd).clone()
se3state_2 = data.read_baxter_se3state_file(datadir+'/se3state2000.txt')
poses_2  = torch.Tensor(1, len(se3state_2)+1, 3, 4).fill_(0) # Setup size
poses_2[0,0,:,0:3] = torch.eye(3).float()  # Identity transform for BG
for id, tfm in se3state_2.items():
    se3tfm = torch.mm(cam_extrinsics['modelView'], tfm)  # NOTE: Do matrix multiply, not * (cmul) here. Camera data is part of options
    poses_2[0][id] = se3tfm[0:3, :] # 3 x 4 transform (id is 1-indexed already, 0 is BG)

# Do bilateral smoothing
depths_1 = data.BilateralDepthSmoothing(depth=depth_1, width=9, depthstd=0.005) # 1mm std.dev
depths_2 = data.BilateralDepthSmoothing(depth=depth_2, width=9, depthstd=0.005) # 1mm std.dev

# Viz
import matplotlib.pyplot as plt
plt.ion()
plt.figure(100); plt.imshow(depth_1.squeeze().numpy())
plt.figure(101); plt.imshow(depths_1.squeeze().numpy())
plt.figure(102); plt.imshow((depth_1 - depths_1).abs().squeeze().numpy())
plt.show()

# Compute point clouds
cloud_1  = se3nn.DepthImageToDense3DPoints(height=ht,width=wd)(util.to_var(depths_1)).data.clone()
cloud_2  = se3nn.DepthImageToDense3DPoints(height=ht,width=wd)(util.to_var(depths_2)).data.clone()

# Compute the flows and visibility
tarpts    = cloud_2    # t+1, t+2, t+3, ....
initpts   = cloud_1
tarlabels = labels_2    # t+1, t+2, t+3, ....
initlabel = labels_1
tarposes  = poses_2  # t+1, t+2, t+3, ....
initpose  = poses_1

# Compute flow and visibility
fwdflows, bwdflows, \
fwdvisibilities, bwdvisibilities = data.ComputeFlowAndVisibility(initpts, tarpts, initlabel, tarlabels,
                                                                 initpose, tarposes, cam_intrinsics,
                                                                 0.01, 5, False)

# Create fake labels
tcloud_2  = cloud_1 + fwdflows
deltas_12 = data.ComposeRtPair(poses_2, data.RtInverse(poses_1))  # Pose_t+1 * Pose_t^-1


# Get memory for output
normals_1  = torch.FloatTensor(1,3,ht,wd).fill_(0)
tnormals_2 = torch.FloatTensor(1,3,ht,wd).fill_(0)

# Compute the normals
se3layers.ComputeNormals_float(cloud_1,
                               cloud_2,
                               labels_1,
                               deltas_12,
                               normals_1,
                               tnormals_2,
                               0.05)

# Save images to disk (cloud and normals)
cloud1   = (cloud_1.squeeze().permute(1,2,0).clone().numpy() * 1e4).astype(np.uint16)
normals1 = (normals_1.squeeze().permute(1,2,0).clone().numpy() * 1e4).astype(np.uint16)
cv2.imwrite("/home/barun/Projects/se3nets-pytorch/test/normaltest/displaynormals/cloud1.png", cloud1)
cv2.imwrite("/home/barun/Projects/se3nets-pytorch/test/normaltest/displaynormals/normals1.png", normals1)

# Save images to disk (cloud and normals)
tcloud2   = (tcloud_2.squeeze().permute(1,2,0).clone().numpy() * 1e4).astype(np.uint16)
tnormals2 = (tnormals_2.squeeze().permute(1,2,0).clone().numpy() * 1e4).astype(np.uint16)
cv2.imwrite("/home/barun/Projects/se3nets-pytorch/test/normaltest/displaynormals/cloud2.png", tcloud2)
cv2.imwrite("/home/barun/Projects/se3nets-pytorch/test/normaltest/displaynormals/normals2.png", tnormals2)

# Wait before quitting
plt.pause(25)

# # Read camera data
# state = data.read_box_state_file(datadir+'/state0.txt')
# campose_w = util.se3quat_to_rt(state['kinect::kinect_camera_depth_optical_frame']['pose'])
# print(campose_w)

'''
f = 589.3664*0.5
a = 1
zF = 100
zN = 0.01
pMatrix = np.array([f/a, 0.0, 0.0,               0.0,
                    0.0, f,   0.0,               0.0,
                    0.0, 0.0, (zF+zN)/(zN-zF),  -1.0,
                    0.0, 0.0, 2.0*zF*zN/(zN-zF), 0.0], np.float32)

# Render these in OpenGL
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

pygame.init()
display = (800,600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

#gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
#glMultMatrixf(pMatrix)
glMultMatrixf(campose_w.squeeze().inverse().numpy())
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    # Plot points and normals
    glBegin(GL_POINTS)
    for r in xrange(0,240,2):
        for c in xrange(0,320,2):
            n = normals_1[0,:,r,c].squeeze()
            p = cloud_1[0,:,r,c].squeeze()
            glNormal3fv((n[0], n[1], n[2]))
            glVertex3fv((p[0], p[1], p[2]))
    glEnd()

    # Plot points and normals
    d = 0.1
    glBegin(GL_LINES)
    for r in xrange(0,240,2):
        for c in xrange(0,320,2):
            n = normals_1[0, :, r, c].squeeze()
            p = cloud_1[0, :, r, c].squeeze()
            glVertex3fv((p[0], p[1], p[2]))
            glVertex3fv((p[0]+d*n[0], p[1]+d*n[1], p[2]+d*n[2]))
    glEnd()

    pygame.display.flip()
    pygame.time.wait(10)
'''