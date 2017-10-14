# Global imports
import os
import sys
sys.path.append("/home/barun/Projects/se3nets-pytorch/")
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import se2nets
import util
from util import AverageMeter, Tee, DataEnumerator

#### Setup options
# Common
import argparse
import options
parser = options.setup_comon_options()

# Loss options
parser.add_argument('--backprop-only-first-delta', action='store_true', default=False,
                    help='Backprop gradients only to the first delta. Switches from using delta-flow-loss to'
                         'full-flow-loss with copied composed deltas if this is set (default: False)')
parser.add_argument('--pt-wt', default=1, type=float,
                    metavar='WT', help='Weight for the 3D point loss - only FWD direction (default: 1)')
parser.add_argument('--use-full-jt-angles', action='store_true', default=False,
                    help='Use angles of all joints as inputs to the networks (default: False)')

# Pivot options
parser.add_argument('--pose-center', default='pred', type=str,
                    metavar='STR',
                    help='Different options for pose center positions: [pred] | predwmaskmean | predwmaskmeannograd')
parser.add_argument('--delta-pivot', default='', type=str,
                    metavar='STR', help='Pivot prediction for the delta-tfm: [] | pred | ptmean | maskmean | '
                                        'maskmeannograd | posecenter')

args = parser.parse_args(args='-c config/alldata/pivot/ours/8se3_wtsharpenr1s0_1seq_normmsesqrt_motionnormloss_maskmean.yaml')
args.cuda       = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ MAIN
# Create logfile to save prints
logfile = open('logfile.txt', 'w')
backup = sys.stdout
sys.stdout = Tee(sys.stdout, logfile)

########################
############ Parse options
# Set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Get default options & camera intrinsics
args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
args.state_labels = []
for k in xrange(len(args.data)):
    load_dir = args.data[k] #args.data.split(',,')[0]
    try:
        # Read from file
        intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
        print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
        if args.se2_data:
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

    # Compute intrinsic grid
    cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                          cam_intrinsics)

    # Compute intrinsics
    cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

    # Get dimensions of ctrl & state
    try:
        statelabels, ctrllabels, trackerlabels = data.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
        print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
    except:
        statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
        ctrllabels = statelabels  # Just use the labels
        trackerlabels = []
        print("Could not read statectrllabels file. Reverting to labels in statelabels file")
    #args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
    #print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
    args.num_ctrl = len(ctrllabels)
    print('Num ctrl: {}'.format(args.num_ctrl))

    # Find the IDs of the controlled joints in the state vector
    # We need this if we have state dimension > ctrl dimension and
    # if we need to choose the vals in the state vector for the control
    ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
    print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

    # Add to list of intrinsics
    args.cam_intrinsics.append(cam_intrinsics)
    args.cam_extrinsics.append(cam_extrinsics)
    args.ctrl_ids.append(ctrlids_in_state)
    args.state_labels.append(statelabels)

# Data noise
if not hasattr(args, "add_noise_data") or (len(args.add_noise_data) == 0):
    args.add_noise_data = [False for k in xrange(len(args.data))] # By default, no noise
else:
    assert(len(args.data) == len(args.add_noise_data))
if hasattr(args, "add_noise") and args.add_noise: # BWDs compatibility
    args.add_noise_data = [True for k in xrange(len(args.data))]

# Get mean/std deviations of dt for the data
if args.mean_dt == 0:
    args.mean_dt = args.step_len * (1.0 / 30.0)
    args.std_dt = 0.005  # +- 10 ms
    print("Using default mean & std.deviation based on the step length. Mean DT: {}, Std DT: {}".format(
        args.mean_dt, args.std_dt))
else:
    exp_mean_dt = (args.step_len * (1.0 / 30.0))
    assert ((args.mean_dt - exp_mean_dt) < 1.0 / 30.0), \
        "Passed in mean dt ({}) is very different from the expected value ({})".format(
            args.mean_dt, exp_mean_dt)  # Make sure that the numbers are reasonable
    print("Using passed in mean & std.deviation values. Mean DT: {}, Std DT: {}".format(
        args.mean_dt, args.std_dt))

# Image suffix
args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

# Read mesh ids and camera data
args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
args.mesh_ids      = args.baxter_labels['meshIds']

# SE3 stuff
assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
assert (args.delta_pivot in ['', 'pred', 'ptmean', 'maskmean', 'maskmeannograd', 'posecenter']),\
    'Unknown delta pivot type: ' + args.delta_pivot
delta_pivot_type = ' Delta pivot type: {}'.format(args.delta_pivot) if (args.delta_pivot != '') else ''
#args.se3_dim       = ctrlnets.get_se3_dimension(args.se3_type)
#args.delta_se3_dim = ctrlnets.get_se3_dimension(args.se3_type, (args.delta_pivot != '')) # Delta SE3 type
print('Predicting {} SE3s of type: {}.{}'.format(args.num_se3, args.se3_type, delta_pivot_type))

# Sequence stuff
print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

# Loss parameters
print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
    args.loss_scale, args.pt_wt, args.consis_wt))

# Weight sharpening stuff
if args.use_wt_sharpening:
    print('Using weight sharpening to encourage binary mask prediction. Start iter: {}, Rate: {}'.format(
        args.sharpen_start_iter, args.sharpen_rate))

# Loss type
delta_loss = ', Penalizing the delta-flow loss per unroll'
norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
print('3D loss type: ' + args.loss_type + norm_motion + delta_loss)

# Wide model
if args.wide_model:
    print('Using a wider network!')

if args.use_jt_angles:
    print("Using Jt angles as input to the pose encoder")

if args.use_jt_angles_trans:
    print("Using Jt angles as input to the transition model")

# DA threshold / winsize
print("Flow/visibility computation. DA threshold: {}, DA winsize: {}".format(args.da_threshold,
                                                                             args.da_winsize))
if args.use_only_da_for_flows:
    print("Computing flows using only data-associations. Flows can only be computed for visible points")
else:
    print("Computing flows using tracker poses. Can get flows for all input points")

########################
############ Load datasets
# Get datasets
if args.reject_left_motion:
    print("Examples where any joint of the left arm moves by > 0.005 radians inter-frame will be discarded. \n"
          "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
if args.reject_right_still:
    print("Examples where no joint of the right arm move by > 0.015 radians inter-frame will be discarded. \n"
          "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
if args.add_noise:
    print("Adding noise to the depths, actual configs & ctrls")
#noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
#                                                  scale_d=True, std_j=0.02) if args.add_noise else None
noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                 defprob=0.005, noisestd=0.005)
valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                           mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                           reject_left_motion=args.reject_left_motion,
                                                           reject_right_still=args.reject_right_still)
baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                     step_len = args.step_len, seq_len = args.seq_len,
                                                     train_per = args.train_per, val_per = args.val_per,
                                                     valid_filter = valid_filter,
                                                     cam_extrinsics=args.cam_extrinsics,
                                                     cam_intrinsics=args.cam_intrinsics,
                                                     ctrl_ids=args.ctrl_ids,
                                                     state_labels=args.state_labels,
                                                     add_noise=args.add_noise_data)
disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                   img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                                   num_ctrl=args.num_ctrl,
                                                                   #num_state=args.num_state,
                                                                   mesh_ids = args.mesh_ids,
                                                                   #ctrl_ids=ctrlids_in_state,
                                                                   #camera_extrinsics = args.cam_extrinsics,
                                                                   #camera_intrinsics = args.cam_intrinsics,
                                                                   compute_bwdflows=args.use_gt_masks,
                                                                   #num_tracker=args.num_tracker,
                                                                   dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                                   use_only_da=args.use_only_da_for_flows,
                                                                   noise_func=noise_func) # Need BWD flows / masks if using GT masks
train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

# Create a data-collater for combining the samples of the data into batches along with some post-processing
if args.evaluate:
    # Load only test loader
    args.imgdisp_freq = 10 * args.disp_freq  # Tensorboard log frequency for the image data
    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    # sampler = torch.utils.data.dataloader.RandomSampler(test_dataset) # Random sampler
    test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, sampler=sampler,
                                                 pin_memory=args.use_pin_memory,
                                                 collate_fn=test_dataset.collate_batch))
else:
    # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
    train_loader = DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                                  collate_fn=train_dataset.collate_batch))
    val_loader = DataEnumerator(util.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                                collate_fn=val_dataset.collate_batch))

#########################################
###################### Test loading data
train_ids, val_ids = [], []
for epoch in range(args.epochs):
    ## Setup avg time & stats:
    train_time, val_time = AverageMeter(), AverageMeter()

    ## Load training data
    for i in xrange(args.train_ipe):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        j, sample = train_loader.next()
        train_ids.append(sample['id'].clone())

        # Measure data loading time
        train_time.update(time.time() - start)

        # Print
        if i % 100 == 0:
            print('Epoch: {}/{}, Train iter: {}/{}, Time: {}/{}'.format(epoch, args.epochs, i, args.train_ipe,
                                                                        train_time.val, train_time.avg))

    ## Load validation data
    for i in xrange(args.val_ipe):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        j, sample = val_loader.next()
        val_ids.append(sample['id'].clone())

        # Measure data loading time
        val_time.update(time.time() - start)

        # Print
        if i % 100 == 0:
            print('Epoch: {}/{}, Val iter: {}/{}, Time: {}/{}'.format(epoch, args.epochs, i, args.val_ipe,
                                                                      val_time.val, val_time.avg))
