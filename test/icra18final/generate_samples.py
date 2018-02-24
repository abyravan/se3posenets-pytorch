# Global imports
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Torch imports
import torch.utils.data
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import data
import util

#### Setup options
# Common
import configargparse

# Loss options
parser = configargparse.ArgumentParser(description='SE3-Pose-Nets Mask rendering')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH', required=True,
                    help='save directory (default: none)')
parser.add_argument('--num-examples', default=100, type=int, metavar='N',
                    help='Num examples to get data for (default: 100)')
pargs = parser.parse_args()

##################
##### Load saved disk network
print("Loading pre-trained network from: {}".format(pargs.checkpoint))
checkpoint = torch.load(pargs.checkpoint)
args       = checkpoint['args']

##### Create data loader stuff
########################
############ Load datasets
# Get datasets
load_color = None
if args.use_xyzrgb:
    load_color = 'rgb'
elif args.use_xyzhue:
    load_color = 'hsv'
if args.reject_left_motion:
    print("Examples where any joint of the left arm moves by > 0.005 radians inter-frame will be discarded. \n"
          "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
if args.reject_right_still:
    print("Examples where no joint of the right arm move by > 0.015 radians inter-frame will be discarded. \n"
          "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
if args.add_noise:
    print("Adding noise to the depths, actual configs & ctrls")
### Box dataset (vs) Other options
if args.box_data:
    print("Box dataset")
    valid_filter, args.mesh_ids = None, None  # No valid filter
    read_seq_func = data.read_box_sequence_from_disk
else:
    print("Baxter dataset")
    valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                                     mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                     reject_left_motion=args.reject_left_motion,
                                                                     reject_right_still=args.reject_right_still)
    read_seq_func = data.read_baxter_sequence_from_disk
### Noise function
# noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
#                                                  scale_d=True, std_j=0.02) if args.add_noise else None
noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                 defprob=0.005, noisestd=0.005)
### Load functions
baxter_data = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                 step_len=args.step_len, seq_len=args.seq_len,
                                                 train_per=args.train_per, val_per=args.val_per,
                                                 valid_filter=valid_filter,
                                                 cam_extrinsics=args.cam_extrinsics,
                                                 cam_intrinsics=args.cam_intrinsics,
                                                 ctrl_ids=args.ctrl_ids,
                                                 state_labels=args.state_labels,
                                                 add_noise=args.add_noise_data)
disk_read_func = lambda d, i: read_seq_func(d, i, img_ht=args.img_ht, img_wd=args.img_wd,
                                            img_scale=args.img_scale, ctrl_type=args.ctrl_type,
                                            num_ctrl=args.num_ctrl,
                                            # num_state=args.num_state,
                                            mesh_ids=args.mesh_ids,
                                            # ctrl_ids=ctrlids_in_state,
                                            # camera_extrinsics = args.cam_extrinsics,
                                            # camera_intrinsics = args.cam_intrinsics,
                                            compute_bwdflows=args.use_gt_masks,
                                            # num_tracker=args.num_tracker,
                                            dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                            use_only_da=args.use_only_da_for_flows,
                                            noise_func=noise_func,
                                            load_color=load_color,
                                            compute_normals=(args.normal_wt > 0),
                                            maxdepthdiff=args.normal_max_depth_diff,
                                            bismooth_depths=args.bilateral_depth_smoothing,
                                            bismooth_width=args.bilateral_window_width,
                                            bismooth_std=args.bilateral_depth_std,
                                            supervised_seg_loss=(
                                            args.seg_wt > 0))  # Need BWD flows / masks if using GT masks
train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
val_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
test_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset),
                                                                   len(test_dataset)))

#### Use examples from test dataset
samples, ids = [], []
for k in range(pargs.num_examples):
    if k%10 == 0:
        print("Loading example: {}/{}".format(k+1,pargs.num_examples))

    # Load data
    id = np.random.randint(0,len(test_dataset))
    sample = test_dataset[id]
    samples.append(sample)
    ids.append(id)

#### Save set of samples
savedata = {'args': args, 'samples':samples, 'ids': id}
util.create_dir(pargs.save_dir)
torch.save(savedata, pargs.save_dir + "/sampledata.pth.tar")
