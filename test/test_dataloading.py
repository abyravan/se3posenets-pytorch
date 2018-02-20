import torch
import cv2
import sys; sys.path.append('/home/barun/Projects/se3nets-pytorch/')
import data as datav
import ctrlnets

## Test it
import argparse
args = argparse.Namespace()
#args.data = ['/home/barun/Projects/se3nets-pytorch/temp/session_2017-9-2_160752/']
args.data = ['/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions_f/',
             '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_wfixjts_5hrs_Feb10_17/postprocessmotionshalf_f/',
             '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint0/postprocessmotions_f/',
             '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint5/postprocessmotions_f/']
#args.data = ['/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/se2data/4link/']
args.img_suffix = 'sub'
args.step_len = 2
args.seq_len = 15
args.train_per = 0.6
args.val_per = 0.15
args.ctrl_type = 'actdiffvel'
args.batch_size = 16
args.use_pin_memory = False
args.num_workers = 6
args.cuda = True
args.se3_type = 'se3aa'
args.pred_pivot = False
args.num_se3 = 8
args.se2_data = False
args.box_data = False

# Get default options & camera intrinsics
args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
args.state_labels = []
for k in range(len(args.data)):
    load_dir = args.data[k]  # args.data.split(',,')[0]
    try:
        # Read from file
        intrinsics = datav.read_intrinsics_file(load_dir + "/intrinsics.txt")
        print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
        if args.se2_data or args.full_res:
            args.img_ht, args.img_wd = int(intrinsics['ht']), int(intrinsics['wd'])
        else:
            args.img_ht, args.img_wd = 240, 320  # All data except SE(2) data is at 240x320 resolution
        args.img_scale = 1.0 / intrinsics['s']  # Scale of the image (use directly from the data)

        # Setup camera intrinsics
        sc = float(args.img_ht) / intrinsics['ht']  # Scale factor for the intrinsics
        cam_intrinsics = {'fx': intrinsics['fx'] * sc,
                          'fy': intrinsics['fy'] * sc,
                          'cx': intrinsics['cx'] * sc,
                          'cy': intrinsics['cy'] * sc}
        print("Scale factor for the intrinsics: {}".format(sc))
    except:
        print("Could not read intrinsics file, reverting to default settings")
        args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
        cam_intrinsics = {'fx': 589.3664541825391 / 2,
                          'fy': 589.3664541825391 / 2,
                          'cx': 320.5 / 2,
                          'cy': 240.5 / 2}
    print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(args.img_ht, args.img_wd,
                                                                                cam_intrinsics['fx'],
                                                                                cam_intrinsics['fy'],
                                                                                cam_intrinsics['cx'],
                                                                                cam_intrinsics['cy']))

    # Compute intrinsic grid & add to list
    cam_intrinsics['xygrid'] = datav.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                          cam_intrinsics)
    args.cam_intrinsics.append(cam_intrinsics)  # Add to list of intrinsics

    ### BOX (vs) BAXTER DATA
    if args.box_data:
        # Get ctrl dimension
        if args.ctrl_type == 'ballposforce':
            args.num_ctrl = 6
        elif args.ctrl_type == 'ballposeforce':
            args.num_ctrl = 10
        elif args.ctrl_type == 'ballposvelforce':
            args.num_ctrl = 9
        elif args.ctrl_type == 'ballposevelforce':
            args.num_ctrl = 13
        else:
            assert False, "Ctrl type unknown: {}".format(args.ctrl_type)
        print('Num ctrl: {}'.format(args.num_ctrl))
    else:
        # Compute extrinsics
        cam_extrinsics = datav.read_cameradata_file(load_dir + '/cameradata.txt')

        # Get dimensions of ctrl & state
        try:
            statelabels, ctrllabels, trackerlabels = datav.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
            print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
        except:
            statelabels = datav.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
            ctrllabels = statelabels  # Just use the labels
            trackerlabels = []
            print("Could not read statectrllabels file. Reverting to labels in statelabels file")
        # args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
        # print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
        args.num_ctrl = len(ctrllabels)
        print('Num ctrl: {}'.format(args.num_ctrl))

        # Find the IDs of the controlled joints in the state vector
        # We need this if we have state dimension > ctrl dimension and
        # if we need to choose the vals in the state vector for the control
        ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
        print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

        # Add to list of intrinsics
        args.cam_extrinsics.append(cam_extrinsics)
        args.ctrl_ids.append(ctrlids_in_state)
        args.state_labels.append(statelabels)

# Data noise
if not hasattr(args, "add_noise_data") or (len(args.add_noise_data) == 0):
    args.add_noise_data = [False for k in range(len(args.data))] # By default, no noise
else:
    assert(len(args.data) == len(args.add_noise_data))
if hasattr(args, "add_noise") and args.add_noise: # BWDs compatibility
    args.add_noise_data = [True for k in range(len(args.data))]

# Get mean/std deviations of dt for the data
args.mean_dt = args.step_len * (1.0 / 30.0)
args.std_dt = 0.005  # +- 10 ms
print("Using default mean & std.deviation based on the step length. Mean DT: {}, Std DT: {}".format(
    args.mean_dt, args.std_dt))

# Image suffix
args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

# Read mesh ids and camera data (for baxter)
args.baxter_labels = datav.read_statelabels_file(args.data[0] + '/statelabels.txt')
args.mesh_ids      = args.baxter_labels['meshIds']

# SE3 stuff
assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat', 'se3aar']), 'Unknown SE3 type: ' + args.se3_type
args.delta_pivot = ''
delta_pivot_type = ' Delta pivot type: {}'.format(args.delta_pivot) if (args.delta_pivot != '') else ''
print('Predicting {} SE3s of type: {}.{}'.format(args.num_se3, args.se3_type, delta_pivot_type))

# Sequence stuff
print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

########################
############ Load datasets
# Get datasets
load_color = None
args.use_xyzrgb = False
args.use_xyzhue = False
args.reject_left_motion, args.reject_right_still = False, False
args.add_noise = False

print("Baxter dataset")
valid_filter = lambda p, n, st, se, slab: datav.valid_data_filter(p, n, st, se, slab,
                                                                 mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                 reject_left_motion=args.reject_left_motion,
                                                                 reject_right_still=args.reject_right_still)
read_seq_func = datav.read_baxter_sequence_from_disk
### Noise function
#noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
#                                                  scale_d=True, std_j=0.02) if args.add_noise else None
noise_func = lambda d: datav.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                 defprob=0.005, noisestd=0.005)
### Load functions
baxter_data     = datav.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                     step_len = args.step_len, seq_len = args.seq_len,
                                                     train_per = args.train_per, val_per = args.val_per,
                                                     valid_filter = valid_filter,
                                                     cam_extrinsics=args.cam_extrinsics,
                                                     cam_intrinsics=args.cam_intrinsics,
                                                     ctrl_ids=args.ctrl_ids,
                                                     state_labels=args.state_labels,
                                                     add_noise=args.add_noise_data)
disk_read_func  = lambda d, i: read_seq_func(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                             img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                             num_ctrl=args.num_ctrl,
                                             #num_state=args.num_state,
                                             mesh_ids = args.mesh_ids,
                                             #ctrl_ids=ctrlids_in_state,
                                             #camera_extrinsics = args.cam_extrinsics,
                                             #camera_intrinsics = args.cam_intrinsics,
                                             compute_bwdflows=True,
                                             #num_tracker=args.num_tracker,
                                             dathreshold=0.01, dawinsize=5,
                                             use_only_da=False,
                                             noise_func=noise_func,
                                             load_color=load_color,
                                             compute_normals=False,
                                             maxdepthdiff=0.1,
                                             bismooth_depths=False,
                                             bismooth_width=7,
                                             bismooth_std=0.02,
                                             supervised_seg_loss=False) # Need BWD flows / masks if using GT masks
train_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
val_dataset   = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
test_dataset  = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

#############
# # Get dimensions of ctrl & state
# try:
#     statelabels, ctrllabels = datav.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
#     print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
# except:
#     statelabels = datav.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
#     ctrllabels = statelabels # Just use the labels
#     print("Could not read statectrllabels file. Reverting to labels in statelabels file")
# args.num_state, args.num_ctrl = len(statelabels), len(ctrllabels)
# print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
#
# # Find the IDs of the controlled joints in the state vector
# # We need this if we have state dimension > ctrl dimension and
# # if we need to choose the vals in the state vector for the control
# ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
# print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1,-1))

# # Compute intrinsic grid
# args.cam_intrinsics['xygrid'] = datav.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
#                                                                            args.cam_intrinsics)
# # Image suffix
# args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
# print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))
#
# # Read mesh ids and camera data
# args.baxter_labels = datav.read_statelabels_file(load_dir + '/statelabels.txt')
# args.mesh_ids      = args.baxter_labels['meshIds']
# args.cam_extrinsics = datav.read_cameradata_file(load_dir + '/cameradata.txt')
#
# # SE3 stuff
# assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
# args.se3_dim = ctrlnets.get_se3_dimension(args.se3_type, args.pred_pivot)
# print('Predicting {} SE3s of type: {}. Dim: {}'.format(args.num_se3, args.se3_type, args.se3_dim))
#
# # Sequence stuff
# print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))
#
# ########################
# ############ Load datasets
# # Get datasets
# baxter_data     = datav.read_recurrent_baxter_dataset(args.data, args.img_suffix,
#                                                      step_len = args.step_len, seq_len = args.seq_len,
#                                                      train_per = args.train_per, val_per = args.val_per)
# disk_read_func  = lambda d, i: datav.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
#                                                                    img_scale = args.img_scale, ctrl_type = args.ctrl_type,
#                                                                    num_ctrl=args.num_ctrl, num_state=args.num_state,
#                                                                    mesh_ids = args.mesh_ids, ctrl_ids=ctrlids_in_state,
#                                                                    camera_extrinsics = args.cam_extrinsics,
#                                                                    camera_intrinsics = args.cam_intrinsics,
#                                                                    compute_bwdflows=True)
# train_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset
# print('Dataset size => Train: {}'.format(len(train_dataset)))

###########
# Display some stuff
sample = train_dataset[20]

pts   = sample['points']
masks = sample['masks']
flows = sample['fwdflows']
print(pts.size(), masks.size(), flows.size())

# import matplotlib.pyplot as plt
# import torchvision
# depthdisp = torchvision.utils.make_grid(sample['points'].narrow(1,2,1).clone(), normalize=True, range=(0.0,3.0))
# fig = plt.figure(100)
# plt.imshow(depthdisp.permute(1,2,0).numpy())
#
# flowdisp = torchvision.utils.make_grid(sample['fwdflows'].clone(), normalize=True, range=(-0.05,0.05))
# fig1 = plt.figure(101)
# plt.imshow(flowdisp.permute(1,2,0).numpy())
#
# maskdisp = torchvision.utils.make_grid(sample['masks'][0:1].view(-1,1,args.img_ht,args.img_wd).float().clone(), normalize=True, range=(0.0,1.0))
# fig2 = plt.figure(102)
# plt.imshow(maskdisp.permute(1,2,0).numpy())
