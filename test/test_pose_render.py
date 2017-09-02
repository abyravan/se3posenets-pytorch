import torch
import cv2
import data

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

## Test it
import argparse
args = argparse.Namespace()
args.data = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions_f/,,'\
            '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_wfixjts_5hrs_Feb10_17/postprocessmotionshalf_f/,,'\
            '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint0/postprocessmotions_f/,,'\
            '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint5/postprocessmotions_f/'
args.img_suffix = 'sub'
args.step_len = 2
args.seq_len = 3
args.train_per = 0.6
args.val_per = 0.15
args.img_ht = 240
args.img_wd = 320
args.img_scale = 1e-4
args.ctrl_type = 'actdiffvel'
args.batch_size = 16
args.use_pin_memory = False
args.num_workers = 6
args.cuda = True

# Read mesh ids and camera data
load_dir = args.data.split(',,')[0]
args.baxter_labels = data.read_statelabels_file(load_dir + '/statelabels.txt')
args.mesh_ids      = args.baxter_labels['meshIds']
args.cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')
args.cam_intrinsics = {'fx': 589.3664541825391/2,
                       'fy': 589.3664541825391/2,
                       'cx': 320.5/2,
                       'cy': 240.5/2}
args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                           args.cam_intrinsics)

# Test
baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                     step_len = args.step_len, seq_len = args.seq_len,
                                                     train_per = args.train_per, val_per = args.val_per)



#############
# Test function 1
disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                   img_scale = args.img_scale, ctrl_type = 'actdiffvel',
                                                                   mesh_ids = args.mesh_ids,
                                                                   camera_extrinsics = args.cam_extrinsics,
                                                                   camera_intrinsics = args.cam_intrinsics,
                                                                   compute_bwdflows=False) # No need for BWD flows
train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset
print('Dataset size => Train: {}'.format(len(train_dataset)))

# Read an example and plot poses
sample = train_dataset[102349]
depths = sample['points'][0][2:]
poses = sample['poses'][0]

### Normalize image
def normalize_img(img, min=-0.01, max=0.01):
    return (img - min) / (max - min)

# Show depths
cvdepths = normalize_img(depths.permute(1,2,0).expand(240,320,3).clone(), min=0, max=3.0)
cv2.imshow("orig", cvdepths.numpy())
for k in xrange(poses.size(0)):
    draw_3d_frame(cvdepths, poses[k], [1,0,0], args.cam_intrinsics)
cv2.imshow("frames", cvdepths.numpy())

