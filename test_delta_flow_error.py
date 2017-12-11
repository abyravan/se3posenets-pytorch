# Global imports
import os
import sys
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
import util.quat as uq

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
                    metavar='STR', help='Different options for pose center positions: [pred] | predwmaskmean | predwmaskmeannograd')
parser.add_argument('--delta-pivot', default='', type=str,
                    metavar='STR', help='Pivot prediction for the delta-tfm: [] | pred | ptmean | maskmean | '
                                        'maskmeannograd | posecenter')
parser.add_argument('--consis-rt-loss', action='store_true', default=False,
                    help='Use RT loss for the consistency measure (default: False)')

# Box data
parser.add_argument('--box-data', action='store_true', default=False,
                    help='Dataset has box/ball data (default: False)')

# GT options
parser.add_argument('--use-gt-poses-wdeltas', action='store_true', default=False,
                    help='Use GT poses along with deltas (default: False)')
parser.add_argument('--use-gt-deltas', action='store_true', default=False,
                    help='Use GT deltas only (default: False)')
parser.add_argument('--use-gt-masks-poses', action='store_true', default=False,
                    help='Use GT masks and poses (default: False)')

# Kinchain options
parser.add_argument('--use-pose-kinchain', action='store_true', default=False,
                    help='Use Kinematic chain structure for the poses only (default: False)')
parser.add_argument('--kinchain-right-to-left', action='store_true', default=False,
                    help='Go from right to left for kinchain computation (default: False)')

# Supervised delta loss
parser.add_argument('--delta-wt', default=0, type=float,
                    metavar='WT', help='Weight for the supervised loss on delta-poses (default: 0)')
parser.add_argument('--delta-rt-loss', action='store_true', default=False,
                    help='Use R/t loss instead of MSE loss (default: False)')
parser.add_argument('--rot-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the supervised loss on delta-poses - rotation (default: 1.0)')
parser.add_argument('--trans-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the supervised loss on delta-poses - translation (default: 1.0)')

# Transition model types
parser.add_argument('--trans-type', default='default', type=str,
                    metavar='STR', help='Different transition model types: [default] | deep | simple | '
                                        'simplewide | simpledense')
parser.add_argument('--trans-bn', action='store_true', default=False,
                    help='Batch Normalize the transition model (default: False)')

# params
parser.add_argument('--num-samples', default=100, type=int,
                    metavar='N', help='Number of examples to test (default: 100)')
parser.add_argument('--num-inter',   default=9, type=int,
                    metavar='N', help='Number of interpolation steps (excluding end points) (default: 9)')
parser.add_argument('--only-trans', action='store_true', default=False,
                    help='Do only translation, no rotation (default: False)')
parser.add_argument('--only-rot', action='store_true', default=False,
                    help='Do only rotation, no translation (default: False)')

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ MAIN
#@profile
def main():
    # Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()
    args.batch_norm = not args.no_batch_norm

    ### Create save directory and start tensorboard logger
    util.create_dir(args.save_dir)  # Create directory
    now = time.strftime("%c")
    tblogger = util.TBLogger(args.save_dir + '/logs/' + now)  # Start tensorboard logger

    # Create logfile to save prints
    logfile = open(args.save_dir + '/logs/' + now + '/logfile.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 480 x 640 or 240 x 320
    if args.full_res:
        print("Using full-resolution images (480x640)")
    # XYZ-RGB
    if args.use_xyzrgb:
        print("Using XYZ-RGB input - 6 channels. Assumes registered depth/RGB")

    # Get default options & camera intrinsics
    args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
    args.state_labels = []
    for k in xrange(len(args.data)):
        load_dir = args.data[k] #args.data.split(',,')[0]
        try:
            # Read from file
            intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
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
        cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                              cam_intrinsics)
        args.cam_intrinsics.append(cam_intrinsics) # Add to list of intrinsics

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

    # Read mesh ids and camera data (for baxter)
    if (not args.box_data):
        args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
        args.mesh_ids      = args.baxter_labels['meshIds']

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
    args.delta_pivot = '' if (args.delta_pivot == 'None') else args.delta_pivot # Workaround since we can't specify empty string in the yaml
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

    # Box data
    if args.box_data:
        assert (not args.use_jt_angles), "Cannot use joint angles as input to the encoder for box data"
        assert (not args.use_jt_angles_trans), "Cannot use joint angles as input to the transition model for box data"
        assert (not args.reject_left_motion), "Cannot filter left arm motions for box data"
        assert (not args.reject_right_still), "Cannot filter right arm still cases for box data"

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
    ### Box dataset (vs) Other options
    if args.box_data:
        print("Box dataset")
        valid_filter, args.mesh_ids = None, None # No valid filter
        read_seq_func = data.read_box_sequence_from_disk
    else:
        print("Baxter dataset")
        valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                                         mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                         reject_left_motion=args.reject_left_motion,
                                                                         reject_right_still=args.reject_right_still)
        read_seq_func = data.read_baxter_sequence_from_disk
    ### Noise function
    #noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
    #                                                  scale_d=True, std_j=0.02) if args.add_noise else None
    noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                     defprob=0.005, noisestd=0.005)
    ### Load functions
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
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
                                                 compute_bwdflows=True, #args.use_gt_masks or args.use_gt_masks_poses,
                                                 #num_tracker=args.num_tracker,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows,
                                                 noise_func=noise_func,
                                                 load_color=args.use_xyzrgb) # Need BWD flows / masks if using GT masks
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))


    ##################
    # Take the test dataset, sample a random number of examples.
    # Compute delta for each example, interpolate in delta space -> 0,0.1,0.2,...,1.0
    # Compute the corresponding transformed point cloud for each
    # Compute delta and flow errors for each, save results
    # Plot mean & std.dev of these errors as we interpolate to final target

    ## Stats array
    stats = argparse.Namespace()
    stats.gtdeltas, stats.deltas               = [], []
    stats.roterr, stats.transerr, stats.abserr = [], [], []
    stats.flowerr_sum, stats.flowerr_avg       = [], []
    stats.motionerr_sum, stats.motionerr_avg   = [], []
    stats.stillerr_sum, stats.stillerr_avg     = [], []
    stats.motion_err, stats.motion_npt, stats.still_err, stats.still_npt = [], [], [], []

    roterr, transerr, abserr   = AverageMeter(), AverageMeter(), AverageMeter()
    flowerrsum, flowerravg     = AverageMeter(), AverageMeter()
    motionerrsum, motionerravg = AverageMeter(), AverageMeter()
    stillerrsum, stillerravg   = AverageMeter(), AverageMeter()

    ## Start run
    num_samples = args.num_samples
    ninter = args.num_inter
    ids, discard_ids = [], []
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    while len(ids) < num_samples:
        # Sample an example from the test dataset
        id = random.randint(0, num_samples-1)
        if id in ids:
            continue
        ids.append(id)

        # Get the example
        sample = test_dataset[id]

        # Interpolate between pose0 & pose1
        #try:
        pose0, pose1 =  sample['poses'][0],  sample['poses'][1] # nSE3 x 3 x 4
        posestart = pose0.unsqueeze(0).expand(ninter+2, pose0.size(0), 3, 4).clone() # Init poses (same for all = pose0)
        poseinter = torch.zeros(ninter+2, pose0.size(0), 3, 4) # Interpolated target poses (Between pose0->pose1)
        for k in xrange(pose0.size(0)):
            rot0, rot1 = pose0[k,:,:-1].clone(), pose1[k,:,:-1].clone()
            if ((rot0 - rot1).abs().max() >= 1e-4): # NO rotation
                poseinter[:,k,:,:-1] = uq.interpolate_rot_matrices(rot0, rot1, ninter, include_endpoints=True) # Rotation
            else:
                poseinter[:,k,:,:-1] = rot0.unsqueeze(0).expand_as(poseinter[:,k,:,:-1]) # Both are same!
            trans0, trans1 = pose0[k,:,-1].clone(), pose1[k,:,-1].clone()
            poseinter[:,k,:,-1] = trans0 + ((trans1 - trans0).view(1,3) * torch.linspace(0,1,ninter+2).view(-1,1))
        # except:
        #     discard_ids.append(id)
        #     ids.remove(id) # Discarded now!
        #     print("Discarded example due to interpolation issues. Num discarded: {}".format(len(discard_ids)))
        #     continue

        # Compute the deltas now (from zero delta to GT delta)
        deltainter = data.ComposeRtPair(poseinter, data.RtInverse(posestart)) # pose1 * pose0^-1  (ninter+2 x nSE3 x 3 x 4)
        deltas     = util.to_var(deltainter.type(deftype), volatile=True)
        if args.only_rot:
            deltas[:,:,:,3] = 0 # No translation
        elif args.only_trans:
            deltas[:,:,:,:3] = torch.eye(3).view(1,1,3,3).expand_as(deltas[:,:,:,:3]).type(deftype)

        # Get the ptcloud, masks
        pts   = util.to_var(sample['points'][0:1].expand(ninter+2,3,args.img_ht,args.img_wd).type(deftype), volatile=True)
        masks = util.to_var(sample['masks'][0:1].expand(ninter+2,pose0.size(0),args.img_ht,args.img_wd).type(deftype))

        # Get GT deltas & flows
        gtfwdflows = util.to_var(sample['fwdflows'][0:1].expand(ninter+2,3,args.img_ht,args.img_wd).type(deftype))
        pose0_g    = util.to_var(sample['poses'][0:1].type(deftype).clone(), requires_grad=False)  # Use GT poses
        pose1_g    = util.to_var(sample['poses'][1:2].type(deftype).clone(), requires_grad=False)  # Use GT poses
        gtdeltas   = se3nn.ComposeRtPair()(pose1_g, se3nn.RtInverse()(pose0_g)).expand_as(deltas).clone() # pose1_g * pose0_g^-1 (GT delta pose)

        # Now predict the points forward based on the interpolated deltas
        predpts   = se3nn.NTfm3D()(pts, masks, deltas) # ninter+2 x 3 x ht x wd

        # Compute errors now!
        # A) Use a loss directly on the delta transforms (supervised)
        # Get rotation and translation error
        delta_diff = se3nn.ComposeRtPair()(deltas, se3nn.RtInverse()(gtdeltas))
        costheta = (0.5 * ((delta_diff[:,:,0,0] + delta_diff[:,:,1,1] + delta_diff[:,:,2,2]) - 1.0))

        # Store physically meaningful errors - rot error in degrees, trans error in cm
        theta     = torch.acos(costheta.clamp(max=1.0)) * (180.0 / 3.14159265359)
        rot_err   = theta.abs().sum(1)  # Sum across the num SE3s
        trans_err = ((deltas[:,:,:,3] - gtdeltas[:,:,:,3]) * 100.0).norm(dim=2, p=2).sum(1)  # Sum across nSE3s
        abs_err   = (deltas - gtdeltas).view(ninter+2,-1).abs().sum(1) # MSE error per example
        stats.roterr.append(rot_err.data.cpu())
        stats.transerr.append(trans_err.data.cpu())
        stats.abserr.append(abs_err.data.cpu())

        # B) Get flow error
        # Compute flow predictions and errors
        # NOTE: I'm using CUDA here to speed up computation by ~4x
        predflows = (predpts - pts).data.unsqueeze(1).clone()
        flows     = gtfwdflows.data.unsqueeze(1).clone()
        flowerr_sum, flowerr_avg, \
            motionerr_sum, motionerr_avg, \
            stillerr_sum, stillerr_avg, \
            motion_err, motion_npt, \
            still_err, still_npt         = compute_masked_flow_errors(predflows, flows)

        # Update stats
        stats.flowerr_sum.append(flowerr_sum); stats.flowerr_avg.append(flowerr_avg)
        stats.motionerr_sum.append(motionerr_sum); stats.motionerr_avg.append(motionerr_avg)
        stats.stillerr_sum.append(stillerr_sum); stats.stillerr_avg.append(stillerr_avg)
        stats.motion_err.append(motion_err); stats.motion_npt.append(motion_npt)
        stats.still_err.append(still_err); stats.still_npt.append(still_npt)

        ######## Plot stuff
        # Avg errors:
        roterr.update(rot_err.data.cpu())
        transerr.update(trans_err.data.cpu())
        abserr.update(abs_err.data.cpu())
        flowerrsum.update(flowerr_sum)
        flowerravg.update(flowerr_avg)
        motionerrsum.update(motionerr_sum)
        motionerravg.update(motionerr_avg)
        stillerrsum.update(stillerr_sum)
        stillerravg.update(stillerr_avg)

        ## Print stuff
        if (len(ids) % args.disp_freq == 0) or (len(ids) == args.num_samples):
            print('Example: {}/{}, Errors: '.format(len(ids), args.num_samples))
            for kk in xrange(ninter+2):
                print('\tJ: {}, FlowS: {:.3f} ({:.3f}), FlowA: {:.4f} ({:.4f}), Rot: {:.4f} ({:.4f}), Trans: {:.4f} ({:.4f}), '
                      'Abs: {:.4f} ({:.4f})'.format(
                    kk,
                    flowerrsum.val[kk], flowerrsum.avg[kk],
                    flowerravg.val[kk], flowerravg.avg[kk],
                    roterr.val[kk], roterr.avg[kk],
                    transerr.val[kk], transerr.avg[kk],
                    abserr.val[kk], abserr.avg[kk]
                ))

    # Save stuff
    save_checkpoint({
        'args': args,
        'stats': stats,
        'ids': ids,
        'discard_ids': discard_ids,
    }, savedir=args.save_dir, filename='stats.pth.tar')

### Compute flow errors for moving / non-moving pts (flows are size: B x S x 3 x H x W)
def compute_masked_flow_errors(predflows, gtflows):
    batch, seq = predflows.size(0), predflows.size(1)  # B x S x 3 x H x W
    # Compute num pts not moving per mask
    # !!!!!!!!! > 1e-3 returns a ByteTensor and if u sum within byte tensors, the max value we can get is 255 !!!!!!!!!
    motionmask = (gtflows.abs().sum(2) > 1e-3).type_as(gtflows)  # B x S x 1 x H x W
    err = (predflows - gtflows).mul_(1e2).pow(2).sum(2)  # B x S x 1 x H x W

    # Compute errors for points that are supposed to move
    motion_err = (err * motionmask).view(batch, seq, -1).sum(2)  # Errors for only those points that are supposed to move
    motion_npt = motionmask.view(batch, seq, -1).sum(2)  # Num points that move (B x S)

    # Compute errors for points that are supposed to not move
    motionmask.eq_(0)  # Mask out points that are not supposed to move
    still_err = (err * motionmask).view(batch, seq, -1).sum(2)  # Errors for non-moving points
    still_npt = motionmask.view(batch, seq, -1).sum(2)  # Num non-moving pts (B x S)

    # Bwds compatibility to old error
    full_err_avg = (motion_err + still_err) / motion_npt
    full_err_avg[full_err_avg != full_err_avg] = 0  # Clear out any Nans
    full_err_avg[full_err_avg == np.inf] = 0  # Clear out any Infs
    full_err_sum, full_err_avg = (motion_err + still_err).sum(1), full_err_avg.sum(1)  # B, B

    # Compute sum/avg stats
    motion_err_avg = (motion_err / motion_npt)
    motion_err_avg[motion_err_avg != motion_err_avg] = 0  # Clear out any Nans
    motion_err_avg[motion_err_avg == np.inf] = 0  # Clear out any Infs
    motion_err_sum, motion_err_avg = motion_err.sum(1), motion_err_avg.sum(1)  # B, B

    # Compute sum/avg stats
    still_err_avg = (still_err / still_npt)
    still_err_avg[still_err_avg != still_err_avg] = 0  # Clear out any Nans
    still_err_avg[still_err_avg == np.inf] = 0  # Clear out any Infs
    still_err_sum, still_err_avg = still_err.sum(1), still_err_avg.sum(1)  # B, B

    # Return
    return full_err_sum.cpu().float(), full_err_avg.cpu().float(), \
           motion_err_sum.cpu().float(), motion_err_avg.cpu().float(), \
           still_err_sum.cpu().float(), still_err_avg.cpu().float(), \
           motion_err.cpu().float(), motion_npt.cpu().float(), \
           still_err.cpu().float(), still_npt.cpu().float()

### Save checkpoint
def save_checkpoint(state, savedir='.', filename='checkpoint.pth.tar'):
    savefile = savedir + '/' + filename
    torch.save(state, savefile)

################ RUN MAIN
if __name__ == '__main__':
    main()
