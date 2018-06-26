# Global imports
import _init_paths
import argparse
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# TODO: Make this cleaner, we don't need most of these parameters to create the pangolin window
img_ht, img_wd, img_scale = 240, 320, 1e-4
seq_len = 1 # For now, only single step
num_se3 = 8 #20 # TODO: Especially this parameter!
dt = 1.0/30.0
oldgrippermodel = False # TODO: When are we actually going to use the new ones?
cam_intrinsics = {'fx': 589.3664541825391 / 2,
                  'fy': 589.3664541825391 / 2,
                  'cx': 320.5 / 2,
                  'cy': 240.5 / 2}
savedir = 'temp' # TODO: Fix this!
posenets = 1 # Using pose nets here!

# Load pangolin visualizer library
from torchviz import simctrlviz
pangolin = simctrlviz.PySimCtrlViz(img_ht, img_wd, img_scale, num_se3,
                                   cam_intrinsics['fx'], cam_intrinsics['fy'],
                                   cam_intrinsics['cx'], cam_intrinsics['cy'],
                                   savedir, posenets)

##########
# NOTE: When importing torch before initializing the pangolin window, I get the error:
#   Pangolin X11: Unable to retrieve framebuffer options
# Long story short, initializing torch before pangolin messes things up big time.
# Also, only the conda version of torch works otherwise there's an issue with loading the torchviz library before torch
#   ImportError: dlopen: cannot load any more object with static TLS
# With the new CUDA & NVIDIA drivers the new conda also doesn't work. had to move to CYTHON to get code to work

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import util
from util import AverageMeter, Tee

# Xalglib
import xalglib

##########
# Parse arguments
parser = argparse.ArgumentParser(description='Reactive control using SE3-Pose-Nets')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for training (default: none)')
parser.add_argument('--args-checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for loading arguments (default: none)')
parser.add_argument('--use-gt-poses-transnet', action='store_true', default=False,
                    help='Use transition model trained directly on GT poses (default: False)')

# Problem options
parser.add_argument('--only-top4-jts', action='store_true', default=False,
                    help='Controlling only the first 4 joints (default: False)')
parser.add_argument('--only-top6-jts', action='store_true', default=False,
                    help='Controlling only the first 6 joints (default: False)')
parser.add_argument('--ctrl-specific-jts', type=str, default='', metavar='JTS',
                    help='Comma separated list of joints to control. All other jts will have 0 error '
                         'but the system can move those (default: '' => all joints are controlled)')

# Planner options
parser.add_argument('--loss-wt', default=1.0, type=float, metavar='EPS',
                    help='Loss scale (default: 1)')
parser.add_argument('--smooth-wt', default=0.0, type=float, metavar='EPS',
                    help='Control smoothness weight (default: 0)')
parser.add_argument('--loss-threshold', default=0, type=float, metavar='EPS',
                    help='Threshold for convergence check based on the losses (default: 0)')

# Misc options
parser.add_argument('--disp-freq', '-p', default=20, type=int,
                    metavar='N', help='print/disp/save frequency (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-configs', type=int, default=-1, metavar='N',
                    help='Num configs to test. <=0 = all (default: all)')

# Display/Save options
parser.add_argument('-s', '--save-dir', default='', type=str,
                    metavar='PATH', help='directory to save results in. (default: <checkpoint_dir>/planlogs/)')
#parser.add_argument('--save-frames', action='store_true', default=False,
#                    help='Enables post-saving of generated frames, very slow process (default: False)')
#parser.add_argument('--save-frame-stats', action='store_true', default=False,
#                    help='Saves all necessary data for genering rendered frames later (default: False)')

# Choose time horizon, optimizer, step length etc
parser.add_argument('--horizon', default=20, type=int, help='Length of the planning horizon')
parser.add_argument('--goal-horizon', default=10, type=int, help='Length of the target horizon')
parser.add_argument('--optimizer', default='sgddirn', type=str,
                    help='Type of optimizer. [sgddirn] | sgd | adam | alglib_lbfgs | alglib_cg')
parser.add_argument('--step-len', default=0.1, type=float,
                    help='Step length scaling the update for each step of the optimization')
parser.add_argument('--max-iter', default=100, type=int, help='Max number of open loop iterations')
parser.add_argument('--ctrl-init', default='zero', type=str, help='Method to initialize control inputs: [zero] | random | gtctrls')

# Dataset to use & num examples to sample
parser.add_argument('--dataset', default='test', type=str,
                    help='Dataset to sample examples from. [test] | train | val')
parser.add_argument('--nexamples', default=5, type=int, help='Number of examples to sample')

def main():
    # Parse args
    global pargs, args, num_train_iter
    pargs = parser.parse_args()
    pargs.cuda = not pargs.no_cuda and torch.cuda.is_available()

    # Create save directory and start tensorboard logger
    if pargs.save_dir == '':
        checkpoint_dir = pargs.checkpoint.rpartition('/')[0]
        pargs.save_dir = checkpoint_dir + '/planlogs/'
    print('Saving planning logs at: ' + pargs.save_dir)
    util.create_dir(pargs.save_dir)  # Create directory
    tblogger = util.TBLogger(pargs.save_dir + '/logs/')  # Start tensorboard logger

    # Create logfile to save prints
    logfile   = open(pargs.save_dir + '/logfile.txt', 'w')
    errorfile = open(pargs.save_dir + '/errorlog.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

    # Set seed
    torch.manual_seed(pargs.seed)
    np.random.seed(pargs.seed)
    if pargs.cuda:
        torch.cuda.manual_seed(pargs.seed)

    # Default tensor type
    deftype = 'torch.cuda.FloatTensor' if pargs.cuda else 'torch.FloatTensor' # Default tensor type
    pargs.deftype = deftype

    ########################
    ############ Load pre-trained network

    # Load data from checkpoint
    # TODO: Print some stats on the training so far, reset best validation loss, best epoch etc
    if os.path.isfile(pargs.checkpoint):
        print("=> loading checkpoint '{}'".format(pargs.checkpoint))
        checkpoint   = torch.load(pargs.checkpoint)
        args         = checkpoint['args']
        try:
            num_train_iter = checkpoint['num_train_iter']
        except:
            num_train_iter = checkpoint['epoch'] * args.train_ipe
        print("=> loaded checkpoint (epoch: {}, num train iter: {})"
              .format(checkpoint['epoch'], num_train_iter))
    else:
        print("=> no checkpoint found at '{}'".format(pargs.checkpoint))
        raise RuntimeError

    # BWDs compatibility
    if pargs.use_gt_poses_transnet:
        checkpoint1 = torch.load(pargs.args_checkpoint) # For loading arguments
        args_1 = checkpoint1['args']
        args.step_len, args.img_suffix = 2, 'sub'
        args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
        args.train_per, args.val_per = 0.6, 0.15
        args.ctrl_type = 'actdiffvel'
        args.cam_intrinsics = args_1.cam_intrinsics
        args.cam_extrinsics = args_1.cam_extrinsics
        args.ctrl_ids       = args_1.ctrl_ids
        args.state_labels   = args_1.state_labels
        args.add_noise_data = args_1.add_noise_data
        args.mesh_ids = args_1.mesh_ids
        args.da_winsize = args_1.da_winsize
        args.da_threshold = args_1.da_threshold
        args.use_only_da_for_flows = args_1.use_only_da_for_flows
    if not hasattr(args, "use_gt_masks"):
        args.use_gt_masks, args.use_gt_poses = False, False
    if not hasattr(args, "num_state"):
        args.num_state = args.num_ctrl
    if not hasattr(args, "use_gt_angles"):
        args.use_gt_angles, args.use_gt_angles_trans = False, False
    if not hasattr(args, "num_state"):
        args.num_state = 7
    if not hasattr(args, "mean_dt"):
        args.mean_dt = args.step_len * (1.0/30.0)
        args.std_dt  = 0.005 # Default params
    if not hasattr(args, "delta_pivot"):
        args.delta_pivot = ''
    if not hasattr(args, "pose_center"):
        args.pose_center = 'pred'

    if not hasattr(args, "use_full_jt_angles"):
        args.use_full_jt_angles = True
    if args.use_full_jt_angles:
        args.num_state_net = args.num_state
    else:
        args.num_state_net = args.num_ctrl

    ## TODO: Either read the args right at the top before calling pangolin - might be easier, somewhat tricky to do BWDs compatibility
    ## TODO: Or allow pangolin to change the args later
    if pargs.use_gt_poses_transnet:
        model = ctrlnets.TransitionModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                         se3_type=args.se3_type, nonlinearity=args.nonlin,
                                         init_se3_iden=args.init_se3_iden, use_kinchain=args.use_kinchain,
                                         delta_pivot='', local_delta_se3=False, use_jt_angles=False)
        posemaskpredfn = None
    else:
        modelfn = ctrlnets.MultiStepSE3OnlyPoseModel if args.use_gt_masks else ctrlnets.MultiStepSE3PoseModel
        model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3, delta_pivot=args.delta_pivot,
                        se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                        input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                        init_posese3_iden=args.init_posese3_iden,
                        init_transse3_iden=args.init_transse3_iden,
                        use_wt_sharpening=args.use_wt_sharpening,
                        sharpen_start_iter=args.sharpen_start_iter,
                        sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                        decomp_model=args.decomp_model, wide=args.wide_model,
                        use_jt_angles=args.use_jt_angles, use_jt_angles_trans=args.use_jt_angles_trans,
                        num_state=args.num_state_net)
        posemaskpredfn = model.forward_only_pose if args.use_gt_masks else model.forward_pose_mask
    if pargs.cuda:
        model.cuda() # Convert to CUDA if enabled

    # Update parameters from trained network
    try:
        model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluate mode
    model.eval()

    # Sanity check some parameters (TODO: remove it later)
    assert(args.num_se3 == num_se3)
    assert(args.img_scale == img_scale)
    try:
        cam_i = args.cam_intrinsics[0] if type(args.cam_intrinsics) is list else args.cam_intrinsics
        for _, key in enumerate(cam_intrinsics):
            assert(cam_intrinsics[key] == cam_i[key])
    except AttributeError:
        args.cam_intrinsics = cam_intrinsics # In case it doesn't exist

    ########################
    ############ Get the data
    # Get datasets (TODO: Make this path variable)
    data_path = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/'
    baxter_data = data.read_recurrent_baxter_dataset(data_path, args.img_suffix,
                                                     step_len = args.step_len, seq_len = pargs.goal_horizon,
                                                     train_per = args.train_per, val_per = args.val_per,
                                                     valid_filter = None,
                                                     cam_extrinsics=args.cam_extrinsics,
                                                     cam_intrinsics=args.cam_intrinsics,
                                                     ctrl_ids=args.ctrl_ids,
                                                     state_labels=args.state_labels,
                                                     add_noise=args.add_noise_data)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                 img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                 num_ctrl=args.num_ctrl,
                                                 mesh_ids = args.mesh_ids,
                                                 compute_bwdflows=args.use_gt_masks,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows) # Need BWD flows / masks if using GT masks
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # Get start & goal samples
    nexamples = pargs.nexamples
    if pargs.dataset == 'train':
        dataset = train_dataset
    elif pargs.dataset == 'val':
        dataset = val_dataset
    elif pargs.dataset == 'test':
        dataset = test_dataset
    else:
        assert(False)

    ###### Get a set of examples
    start_angles_all, goal_angles_all = torch.zeros(nexamples, args.num_ctrl), torch.zeros(nexamples, args.num_ctrl)
    gt_ctrls_all, gt_angles_all = torch.zeros(nexamples, pargs.goal_horizon, args.num_ctrl), \
                                  torch.zeros(nexamples, pargs.goal_horizon+1, args.num_ctrl)
    start_poses_all, goal_poses_all = torch.zeros(nexamples, args.num_se3, 3, 4), \
                                      torch.zeros(nexamples, args.num_se3, 3, 4)
    gt_poses_all = torch.zeros(nexamples, pargs.goal_horizon+1, args.num_se3, 3, 4)
    assert(pargs.horizon >= pargs.goal_horizon)
    k = 0
    while (k < nexamples):
        # Get examples
        start_id = np.random.randint(len(dataset))
        print('Test dataset size: {}, Start ID: {}, Duration: {}'.format(len(dataset),
                          start_id, pargs.goal_horizon * args.step_len * dt))
        sample = dataset[start_id]

        # Get joint angles and check if there is reasonable motion
        poses, jtangles, ctrls = sample['poses'], sample['actctrlconfigs'], sample['controls']

        # Test poses first
        st_poses = generate_poses(jtangles[0], args.mesh_ids, args.cam_extrinsics[0]['modelView'])
        gl_poses = generate_poses(jtangles[-1], args.mesh_ids, args.cam_extrinsics[0]['modelView'])
        print('ST', (st_poses[0] - poses[0]).abs().mean(), (st_poses[0] - poses[0]).abs().max())
        print('GL', (gl_poses[0] - poses[-1]).abs().mean(), (gl_poses[0] - poses[-1]).abs().max())

        if (jtangles[0] - jtangles[-1]).abs().mean() < (pargs.goal_horizon * 0.02):
            continue
        print('Example: {}/{}, Mean motion between start & goal is {} > {}'.format(k+1, nexamples,
            (jtangles[0] - jtangles[-1]).abs().mean(), pargs.goal_horizon * 0.02))

        # Save stuff
        start_angles_all[k], goal_angles_all[k] = jtangles[0], jtangles[-1]
        gt_angles_all[k], gt_ctrls_all[k] = jtangles, ctrls
        start_poses_all[k], goal_poses_all[k] = poses[0], poses[-1]
        gt_poses_all[k] = poses

        # Increment counter
        k += 1

    # #########
    # # SOME TEST CONFIGS:
    # # TODO: Get more challenging test configs by moving arm physically
    # start_angles_all = torch.FloatTensor(
    #     [
    #      [6.5207e-01, -2.6608e-01,  8.2490e-01,  1.0400e+00, -2.9203e-01,  2.1293e-01, 1.1197e-06],
    #      [0.1800, 0.4698, 1.5043, 0.5696, 0.1862, 0.8182, 0.0126],
    #      [1.0549, -0.1554, 1.2620, 1.0577, 1.0449, -1.2097, -0.6803],
    #      [-1.2341e-01, 7.4693e-01, 1.4739e+00, 1.6523e+00, -2.6991e-01, 1.1523e-02, -9.5822e-05],
    #      [-0.5728, 0.6794, 1.4149, 1.7189, -0.6503, 0.3657, -1.6146],
    #      [0.5426, 0.4880, 1.4143, 1.2573, -0.4632, -1.0516, 0.1703],
    #      [0.4412, -0.5150, 0.8153, 1.5142, 0.4762, 0.0438, 0.7105],
    #      [0.0619, 0.1619, 1.1609, 0.9808, 0.3923, 0.6253, 0.0328],
    #      [0.5052, -0.4135, 1.0945, 1.6024, 1.0821, -0.6957, -0.2535],
    #      [-0.6730, 0.5814, 1.3403, 1.7309, -0.4106, 0.4301, -1.7868],
    #      [0.3525, -0.1269, 1.1656, 1.3804, 0.1220, 0.3742, -0.1250]
    #     ]
    #     )
    # goal_angles_all  = torch.FloatTensor(
    #     [
    #      [0.5793, -0.0749,  0.9222,  1.4660, -0.7369, -0.6797, -0.4747],
    #      [0.1206, 0.2163, 1.2128, 0.9753, 0.2447, 0.5462, -0.2298],
    #      [0.0411, -0.4383, 1.1090, 1.9053, 0.7874, -0.1648, -0.3210],
    #      [8.3634e-01, -3.7185e-01, 5.8938e-01, 1.0404e+00, 3.0321e-01, 1.6204e-01, -1.9278e-05],
    #      [-0.5702, 0.6332, 1.4110, 1.6701, -0.5085, 0.4071, -1.6792],
    #      [0.0338, 0.0829, 1.0422, 1.6009, -0.7885, -0.5373, 0.1593],
    #      [0.2692, -0.4469, 0.6287, 0.8841, 0.2070, 1.3161, 0.4913],
    #      [4.1391e-01, -4.5127e-01, 8.9605e-01, 1.1968e+00, -4.4754e-05, 8.8374e-01, 6.2656e-02],
    #      [0.0880, -0.3266, 0.8092, 1.1611, 0.2845, 0.5481, -0.4666],
    #      [-0.0374, -0.2891, 1.2771, 1.4422, -0.4017, 0.9142, -0.7823],
    #      [0.4959, -0.2184, 1.2100, 1.8197, 0.3975, -0.7801, 0.2076]
    #     ]
    #     )

    # Iterate over test configs
    print("Running tests over {} configs".format(nexamples))
    iterstats = []
    init_errors, final_errors = [], []
    datastats = []
    for k in xrange(nexamples):
        # Get start/goal angles
        print("========================================")
        print("========== STARTING TEST: {} ===========".format(k))
        start_angles = start_angles_all[k].clone()
        goal_angles  = goal_angles_all[k].clone()

        # Set based on options
        if pargs.only_top4_jts:
            assert not pargs.only_top6_jts, "Cannot enable control for 4 and 6 joints at the same time"
            print('Controlling only top 4 joints')
            goal_angles[4:] = start_angles[4:]
        elif pargs.only_top6_jts:
            assert not pargs.only_top4_jts, "Cannot enable control for 4 and 6 joints at the same time"
            print('Controlling only top 6 joints')
            goal_angles[6:] = start_angles[6:]
        elif pargs.ctrl_specific_jts is not '':
            print('Setting targets only for joints: {}. All other joints have zero error'
                  ' but can be controlled'.format(pargs.ctrl_specific_jts))
            ctrl_jts = [int(x) for x in pargs.ctrl_specific_jts.split(',')]
            for jj in xrange(7):
                if jj not in ctrl_jts:
                    goal_angles[jj] = start_angles[jj]

        ########################
        ############ Get start & goal point clouds, predict poses & masks

        # Get full goal point cloud
        start_pts, _ = generate_ptcloud(start_angles)
        goal_pts, _  = generate_ptcloud(goal_angles)

        #### Predict start/goal poses and masks
        print('Predicting start/goal poses and masks')
        sinp = [util.to_var(start_pts.type(deftype)), util.to_var(start_angles.view(1, -1).type(deftype))]
        tinp = [util.to_var(goal_pts.type(deftype)), util.to_var(goal_angles.view(1, -1).type(deftype))]

        if pargs.use_gt_poses_transnet:
            start_poses = util.to_var(start_poses_all[k:k+1].clone().type(deftype))
            goal_poses = util.to_var(goal_poses_all[k:k+1].clone().type(deftype))
            _, start_rlabels = generate_ptcloud(start_angles)
            _, goal_rlabels = generate_ptcloud(goal_angles)
            start_masks = util.to_var(compute_masks_from_labels(start_rlabels, args.mesh_ids).type(deftype))
            goal_masks = util.to_var(compute_masks_from_labels(goal_rlabels, args.mesh_ids).type(deftype))
        else:
            # GT masks or both poses and masks
            if args.use_gt_masks: # GT masks are provided!
                _, start_rlabels = generate_ptcloud(start_angles)
                _, goal_rlabels  = generate_ptcloud(goal_angles)
                start_masks = util.to_var(compute_masks_from_labels(start_rlabels, args.mesh_ids).type(deftype))
                goal_masks  = util.to_var(compute_masks_from_labels(goal_rlabels, args.mesh_ids).type(deftype))
                start_poses = posemaskpredfn(sinp)
                goal_poses  = posemaskpredfn(tinp)
            else:
                start_poses, start_masks = posemaskpredfn(sinp, train_iter=num_train_iter)
                goal_poses, goal_masks   = posemaskpredfn(tinp, train_iter=num_train_iter)

            # Update poses if there is a center option
            start_poses, _, _ = ctrlnets.update_pose_centers(sinp[0], start_masks, start_poses, args.pose_center)
            goal_poses, _, _  = ctrlnets.update_pose_centers(tinp[0], goal_masks, goal_poses, args.pose_center)

        # Display the masks as an image summary
        maskdisp = torchvision.utils.make_grid(torch.cat([start_masks.data, goal_masks.data],
                                                         0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                                               nrow=args.num_se3, normalize=True, range=(0, 1))
        info = {'start/goal masks': util.to_np(maskdisp.narrow(0, 0, 1))}
        for tag, images in info.items():
            tblogger.image_summary(tag, images, 0)

        # Init controls
        if pargs.ctrl_init == 'zero':
            ctrls_t = torch.zeros(pargs.horizon, args.num_ctrl)
        elif pargs.ctrl_init == 'random':
            ctrls_t = torch.zeros(pargs.horizon, args.num_ctrl).uniform_(-0.5, 0.5)
        elif pargs.ctrl_init == 'gtctrls':
            ctrls_t = torch.zeros(pargs.horizon, args.num_ctrl)
            ctrls_t[0:pargs.goal_horizon] = gt_ctrls_all[k]
        else:
            assert(False)

        # Init stuff
        ctrls, losses = [ctrls_t.clone()], []
        angles, deg_errors = [torch.zeros(pargs.horizon + 1, args.num_ctrl)], \
                             [torch.zeros(pargs.horizon + 1, args.num_ctrl)]
        angles[0] = integrate_ctrls(start_angles, ctrls_t, dt*args.step_len, pargs)
        deg_errors[0] = (angles[0] - goal_angles.view(1,7)) * (180.0 / np.pi)

        print("Jt angle error using initial controls: ")
        init_deg_error = deg_errors[0][-1]  # Full error in degrees
        total_deg_error = (start_angles - goal_angles) * (180.0 / np.pi)
        print(torch.cat([init_deg_error.view(7,1), total_deg_error.view(7,1)], 1))

        # Compute initial pose loss
        init_pose_loss = pargs.loss_wt * ctrlnets.BiMSELoss(start_poses, goal_poses)  # Get distance from goal
        init_pose_err_indiv = pargs.loss_wt * (start_poses - goal_poses).pow(2).view(args.num_se3,12).mean(1)
        init_deg_errors = init_deg_error.view(7).numpy()

        # Render the poses
        # NOTE: Data passed into cpp library needs to be assigned to specific vars, not created on the fly (else gc will free it)
        print("Initializing the visualizer...")
        start_masks_f, goal_masks_f = start_masks.data.cpu().float(), goal_masks.data.cpu().float()
        start_poses_f, goal_poses_f = start_poses.data.cpu().float(), goal_poses.data.cpu().float()
        pangolin.update_init(start_angles.numpy(),
                             start_pts[0].cpu().numpy(),
                             start_poses_f[0].numpy(),
                             start_masks_f[0].numpy(),
                             #start_rgb.numpy(),
                             goal_angles.numpy(),
                             goal_pts[0].cpu().numpy(),
                             goal_poses_f[0].numpy(),
                             goal_masks_f[0].numpy(),
                             #goal_rgb.numpy(),
                             init_pose_loss.data[0],
                             init_pose_err_indiv.data.cpu().numpy(),
                             init_deg_errors)

        # # For saving:
        # if pargs.save_frame_stats:
        #     initstats = [start_angles.numpy(),
        #                  start_pts[0].cpu().numpy(),
        #                  start_poses_f[0].numpy(),
        #                  start_masks_f[0].numpy(),
        #                  #start_rgb.numpy(),
        #                  goal_angles.numpy(),
        #                  goal_pts[0].cpu().numpy(),
        #                  goal_poses_f[0].numpy(),
        #                  goal_masks_f[0].numpy(),
        #                  #goal_rgb.numpy(),
        #                  init_pose_loss.data[0],
        #                  init_pose_err_indiv.data.cpu().numpy(),
        #                  init_deg_errors]

        ########################
        ############ Run the controller

        # # Plots for errors and loss
        # fig, axes = plt.subplots(2, 1)
        # fig.show()

        # # Save for final frame saving
        # if pargs.save_frames or pargs.save_frame_stats:
        #     curr_angles_s, curr_pts_s, curr_poses_s, curr_masks_s = [], [], [] ,[]
        #     #curr_rgb_s = []
        #     loss_s, err_indiv_s, curr_deg_errors_s = [], [], []

        # Model has to be in training mode
        model.train()
        transmodel = model if pargs.use_gt_poses_transnet else model.transitionmodel

        ############
        # Xalglib optimizers
        pargs.curr_iter = 0
        if pargs.optimizer.find('xalglib') != -1:
            # Setup params
            ctrlstate = list(ctrls_t.view(-1).clone().numpy())
            epsg, epsf, epsx, maxits = 1e-4, 0, 0, pargs.max_iter

            # Create function handle for optimization
            func = lambda x, y, z: optimize_ctrl(x, y, z, transmodel, start_poses, goal_poses,
                                                 start_angles, goal_angles, angles[0])

            # Run the optimization
            if pargs.optimizer == 'xalglib_cg':
                # Setup CG optimizer
                state = xalglib.mincgcreate(ctrlstate)
                pargs.xalglib_optim_state = state
                xalglib.mincgsetcond(state, epsg, epsf, epsx, maxits)
                xalglib.mincgoptimize_g(state, func)
                ctrlstate, rep = xalglib.mincgresults(state)
            elif pargs.optimizer == 'xalglib_lbfgs':
                # Setup CG optimizer
                state = xalglib.minlbfgscreate(5, ctrlstate)
                pargs.xalglib_optim_state = state
                xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
                xalglib.minlbfgsoptimize_g(state, func)
                ctrlstate, rep = xalglib.minlbfgsresults(state)
            else:
                assert(False)

            # Print results
            # Apply controls (simple velocity integration to start joint angles)
            ctrls_f = torch.FloatTensor(ctrlstate).view(pargs.horizon, -1).clone()
            angle_traj = integrate_ctrls(start_angles, ctrls_f, dt*args.step_len, pargs)
            pargs.xalglib_optim_state = None

            ## Print final trajectory errors
            deg_error = (angle_traj - goal_angles.view(1,7)) * (180.0 / np.pi)
            print("Test: {}/{}, Final trajectory errors after {} iterations".format(k + 1, nexamples, pargs.max_iter))
            print(deg_error)
            print(ctrls_f)

            # Save final errors
            deg_errors.append((angle_traj - goal_angles.view(1, 7)) * (180.0 / np.pi))
            angles.append(angle_traj)
            ctrls.append(ctrls_f.clone())

            # Save stats and exit
            iterstats.append({'start_angles': start_angles, 'goal_angles': goal_angles,
                              'start_poses': start_poses.data.cpu(), 'goal_poses': goal_poses.data.cpu(),
                              'angles': angles, 'ctrls': ctrls_f, 'init_ctrls': ctrls_t,
                              'deg_errors': deg_errors, 'losses': losses})

        else:
            # Run open loop planner for a fixed number of iterations
            # This runs only in the pose space!
            for it in range(pargs.max_iter):
                # Print
                print('\n #####################')

                # ============ FWD pass + Compute loss ============#
                # Roll out over the horizon
                start_pose_v = Variable(start_poses.data.clone(), requires_grad=False)
                goal_pose_v  = Variable(goal_poses.data.clone(),  requires_grad=False)
                ctrls_v      = Variable(ctrls_t.clone().type(deftype),       requires_grad=True)
                pred_poses_v = []
                iter_loss   = torch.zeros(pargs.horizon)
                curr_loss_v = 0
                for h in xrange(pargs.horizon):
                    # Get inputs for the current timestep
                    pose = start_pose_v if (h == 0) else pred_poses_v[-1]
                    ctrl = ctrls_v.narrow(0,h,1)

                    # Run a forward pass through the network
                    _, pred_pose = transmodel([pose, ctrl])
                    pred_poses_v.append(pred_pose)

                    # Compute loss and add to list of losses
                    if h > 0.75*pargs.horizon:
                        loss = pargs.loss_wt * ctrlnets.BiMSELoss(pred_pose, goal_pose_v)
                        curr_loss_v += loss
                        iter_loss[h] = loss.data[0]

                # ============ BWD pass ============#

                # Backward pass & optimize
                transmodel.zero_grad()  # Zero gradients
                zero_gradients(ctrls_v)  # Zero gradients for controls
                curr_loss_v.backward()  # Compute gradients - BWD pass
                ctrl_grad = ctrls_v.grad.data.cpu().float()

                # ============ Update controls ============#
                # Get the control direction and scale it by step length
                if pargs.optimizer == 'sgd':
                    ctrls_t = ctrls_t - ctrl_grad * pargs.step_len
                elif pargs.optimizer == 'sgddirn':
                    ctrl_grad_dirn = ctrl_grad / ctrl_grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12).expand_as(ctrl_grad)  # Dirn
                    ctrls_t = ctrls_t - ctrl_grad_dirn * pargs.step_len
                else:
                    assert(False)

                # ============ Print loss ============#
                # Apply controls (simple velocity integration to start joint angles)
                angle_traj = integrate_ctrls(start_angles, ctrls_t, dt*args.step_len, pargs)

                # Save stuff
                ctrls.append(ctrls_t.clone())
                losses.append(iter_loss)
                angles.append(angle_traj)
                deg_errors.append((angle_traj - goal_angles.view(1,7)) * (180.0 / np.pi))

                # Print losses and errors
                print('Test: {}/{}, Control Iter: {}/{}, Loss: {}'.format(k+1, nexamples, it+1,
                                                                          pargs.max_iter, curr_loss_v.data[0]))
                print('Joint angle errors in degrees (curr/init): ',
                      torch.cat([deg_errors[-1][-1].view(7, 1), init_deg_error.unsqueeze(1), total_deg_error.view(7,1)], 1))


            ## Print final trajectory errors
            print("Test: {}/{}, Final trajectory errors after {} iterations".format(k+1, nexamples, pargs.max_iter))
            print(deg_errors[-1])
            print(ctrls[-1])

            # Save stats and exit
            iterstats.append({'start_angles': start_angles, 'goal_angles': goal_angles,
                              'start_poses': start_poses.data.cpu(), 'goal_poses': goal_poses.data.cpu(),
                              'angles': angles, 'ctrls': ctrls,
                              'deg_errors': deg_errors, 'losses': losses})

    # Print all results
    final_deg_errors = torch.zeros(len(iterstats), 7)
    for k in xrange(len(iterstats)):
        final_deg_errors[k] = iterstats[k]['deg_errors'][-1][-1]
    print("Final jt angle errors (deg) after integrating the optimized controls: ")
    print(final_deg_errors)

    # Save stats across all iterations
    stats = {'args': args, 'pargs': pargs, 'start_angles_all': start_angles_all,
             'goal_angles_all': goal_angles_all, 'iterstats': iterstats, 'finalerrors': final_errors}
    if pargs.use_gt_poses_transnet:
        stats['start_poses_all'] = start_poses_all
        stats['goal_poses_all']  = goal_poses_all
    torch.save(stats, pargs.save_dir + '/planstats.pth.tar')
    sys.stdout = backup
    logfile.close()
    errorfile.close()

        # # Run the controller for max_iter iterations
        # conv_iter = pargs.max_iter
        # inc_ctr, max_ctr = 0, 100 # Num consecutive times we've seen an increase in loss
        # for it in xrange(pargs.max_iter):
        #
        #
        #     # Get current point cloud
        #     start = time.time()
        #     curr_angles = angles[it]
        #     curr_pts, curr_rlabels = generate_ptcloud(curr_angles)
        #     curr_pts = curr_pts.type(deftype)
        #     gen_time.update(time.time() - start)
        #
        #     # Predict poses and masks
        #     start = time.time()
        #     #if args.use_jt_angles or args.seq_len > 1:
        #     inp = [util.to_var(curr_pts), util.to_var(curr_angles.view(1, -1).type(deftype))]
        #     #else:
        #     #    inp = util.to_var(curr_pts)
        #     if args.use_gt_masks:
        #         curr_masks = util.to_var(compute_masks_from_labels(curr_rlabels, args.mesh_ids).type(deftype))
        #         curr_poses = posemaskpredfn(inp)
        #     else:
        #         curr_poses, curr_masks = posemaskpredfn(inp, train_iter=num_train_iter)
        #
        #     # Update poses (based on pivots)
        #     curr_poses, _, _ = ctrlnets.update_pose_centers(util.to_var(curr_pts), curr_masks, curr_poses, args.pose_center)
        #
        #     # Get CPU stuff
        #     curr_poses_f, curr_masks_f = curr_poses.data.cpu().float(), curr_masks.data.cpu().float()
        #     #curr_rgb = start_rgb
        #     posemask_time.update(time.time() - start)
        #
        #     # Compute pivots
        #     curr_pivots = None
        #     if not ((args.delta_pivot == '') or (args.delta_pivot == 'pred')):
        #         curr_pivots = ctrlnets.compute_pivots(util.to_var(curr_pts), curr_masks, curr_poses, args.delta_pivot)
        #
        #     # Render poses and masks using Pangolin
        #     start = time.time()
        #     #_, curr_labels = curr_masks_f.max(dim=1)
        #     #curr_labels_f = curr_labels.float()
        #     #pangolin.update_masklabels_and_poses(curr_labels_f.numpy(), curr_poses_f[0].numpy())
        #
        #     # Show masks using tensor flow
        #     if (it % args.disp_freq) == 0:
        #         maskdisp = torchvision.utils.make_grid(curr_masks.data.cpu().view(-1, 1, args.img_ht, args.img_wd),
        #                                                    nrow=args.num_se3, normalize=True, range=(0, 1))
        #         info = {'curr masks': util.to_np(maskdisp.narrow(0, 0, 1))}
        #         for tag, images in info.items():
        #             tblogger.image_summary(tag, images, it)
        #     viz_time.update(time.time() - start)
        #
        #     # Run one step of the optimization (controls are always zeros, poses change)
        #     start = time.time()
        #     ctrl_grad, loss = optimize_ctrl(model=model.transitionmodel,
        #                                     poses=curr_poses, ctrl=init_ctrl_v,
        #                                     angles=angles[it].view(1,-1),
        #                                     goal_poses=goal_poses_v,
        #                                     pivots=curr_pivots)
        #     optim_time.update(time.time() - start)
        #     ctrl_grads.append(ctrl_grad.cpu().float()) # Save this
        #
        #     # Set last 3 joint's controls to zero
        #     if pargs.only_top4_jts:
        #         ctrl_grad[4:] = 0
        #     elif pargs.only_top6_jts:
        #         ctrl_grad[6:] = 0
        #
        #     # Decay controls if loss is small
        #     if loss < 1:
        #         ctrl_mag_decay = pargs.ctrl_mag_decay
        #     else:
        #         ctrl_mag = pargs.max_ctrl_mag
        #         ctrl_mag_decay = 1.
        #
        #     # Get the control direction and scale it by max control magnitude
        #     start = time.time()
        #     if ctrl_mag > 0:
        #         ctrl_dirn = ctrl_grad.cpu().float() / ctrl_grad.norm(2) # Dirn
        #         curr_ctrl = ctrl_dirn * ctrl_mag # Scale dirn by mag
        #         ctrl_mag *= ctrl_mag_decay # Decay control magnitude
        #     else:
        #         curr_ctrl = ctrl_grad.cpu().float()
        #
        #     # Apply control (simple velocity integration)
        #     next_angles = curr_angles - (curr_ctrl * dt)
        #
        #     # Save stuff
        #     losses.append(loss)
        #     ctrls.append(curr_ctrl)
        #     angles.append(next_angles)
        #     deg_errors.append((next_angles-goal_angles).view(1,7)*(180.0/np.pi))
        #
        #     # Print losses and errors
        #     print('Test: {}/{}, Ctr: {}/{}, Control Iter: {}/{}, Loss: {}'.format(k+1, num_configs, inc_ctr, max_ctr,
        #                                                                           it+1, pargs.max_iter, loss))
        #     print('Joint angle errors in degrees: ',
        #           torch.cat([deg_errors[-1].view(7,1), full_deg_error.unsqueeze(1)], 1))
        #
        #     #### Render stuff
        #     curr_pts_f = curr_pts.cpu().float()
        #     curr_deg_errors = (next_angles - goal_angles).view(1, 7) * (180.0 / np.pi)
        #     curr_pose_err_indiv = args.loss_scale * (curr_poses - goal_poses).pow(2).view(args.num_se3, 12).mean(1)
        #     pangolin.update_curr(curr_angles.numpy(),
        #                          curr_pts_f[0].numpy(),
        #                          curr_poses_f[0].numpy(),
        #                          curr_masks_f[0].numpy(),
        #                          #curr_rgb.numpy(),
        #                          loss,
        #                          curr_pose_err_indiv.data.cpu().numpy(),
        #                          curr_deg_errors[0].numpy(),
        #                          0)  # Don't save frame
        #
        #     # Save for future frame generation!
        #     if pargs.save_frames or pargs.save_frame_stats:
        #         curr_angles_s.append(curr_angles)
        #         curr_pts_s.append(curr_pts_f[0])
        #         curr_poses_s.append(curr_poses_f[0])
        #         curr_masks_s.append(curr_masks_f[0])
        #         #curr_rgb_s.append(curr_rgb)
        #         loss_s.append(loss)
        #         err_indiv_s.append(curr_pose_err_indiv.data.cpu())
        #         curr_deg_errors_s.append(curr_deg_errors[0])
        #
        #     # # Plot the errors & loss
        #     # colors = ['r', 'g', 'b', 'c', 'y', 'k', 'm']
        #     # labels = []
        #     # if ((it % 4) == 0) or (loss < pargs.loss_threshold):
        #     #     conv = "Converged" if (loss < pargs.loss_threshold) else ""
        #     #     axes[0].set_title(("Test: {}/{}, Iter: {}, Loss: {}, Jt angle errors".format(k+1, num_configs,
        #     #                                                                                  (it + 1), loss)) + conv)
        #     #     for j in xrange(7):
        #     #         axes[0].plot(torch.cat(deg_errors, 0).numpy()[:, j], color=colors[j])
        #     #         labels.append("Jt-{}".format(j))
        #     #     # I'm basically just demonstrating several different legend options here...
        #     #     axes[0].legend(labels, ncol=4, loc='upper center',
        #     #                    bbox_to_anchor=[0.5, 1.1],
        #     #                    columnspacing=1.0, labelspacing=0.0,
        #     #                    handletextpad=0.0, handlelength=1.5,
        #     #                    fancybox=True, shadow=True)
        #     #     axes[1].set_title("Iter: {}, Loss".format(it + 1))
        #     #     axes[1].plot(losses, color='k')
        #     #     fig.canvas.draw()  # Render
        #     #     plt.pause(0.01)
        #     # if (it % args.disp_freq) == 0: # Clear now and then
        #     #     for ax in axes:
        #     #         ax.cla()
        #
        #     # Finish
        #     rest_time.update(time.time() - start)
        #     print('Gen: {:.3f}({:.3f}), PoseMask: {:.3f}({:.3f}), Viz: {:.3f}({:.3f}),'
        #           ' Optim: {:.3f}({:.3f}), Rest: {:.3f}({:.3f})'.format(
        #         gen_time.val, gen_time.avg, posemask_time.val, posemask_time.avg,
        #         viz_time.val, viz_time.avg, optim_time.val, optim_time.avg,
        #         rest_time.val, rest_time.avg))
        #
        #     # Check for convergence in pose space
        #     if loss < pargs.loss_threshold:
        #         print("*****************************************")
        #         print('Control Iter: {}/{}, Loss: {} less than threshold'.format(it + 1, pargs.max_iter,
        #                                                                          loss, pargs.loss_threshold))
        #         conv_iter = it+1
        #         print("*****************************************")
        #         break
        #
        # # Print final stats
        # print('=========== FINISHED ============')
        # print('Final loss after {} iterations: {}'.format(pargs.max_iter, losses[-1]))
        # print('Final angle errors in degrees: ')
        # print(deg_errors[-1].view(7,1))
        # final_errors.append(deg_errors[-1])
        #
        # # Print final error in stats file
        # if status == -1:
        #     conv = "Final Loss: {}, Failed after {} iterations \n".format(losses[-1], conv_iter)
        # elif conv_iter == pargs.max_iter:
        #     conv = "Final Loss: {}, Did not converge after {} iterations\n".format(losses[-1], pargs.max_iter)
        # else:
        #     conv = "Final Loss: {}, Converged after {} iterations\n".format(losses[-1], conv_iter)
        # errorfile.write(("Test: {}, ".format(k+1)) + conv)
        # for j in xrange(len(start_angles)):
        #     errorfile.write("{}, {}, {}, {}\n".format(start_angles[j], goal_angles[j],
        #                                             full_deg_error[j], deg_errors[-1][0,j]))
        # errorfile.write("\n")

        # # Save
        # if pargs.save_frame_stats:
        #     datastats.append([initstats, curr_angles_s, curr_pts_s, curr_poses_s,
        #                       curr_masks_s, loss_s, err_indiv_s, curr_deg_errors_s]) # curr_rgb_s

        # ###################### RE-RUN VIZ TO SAVE FRAMES TO DISK CORRECTLY
        # ######## Saving frames to disk now!
        # if pargs.save_frames:
        #     pangolin.update_init(start_angles.numpy(),
        #                          start_pts[0].cpu().numpy(),
        #                          start_poses_f[0].numpy(),
        #                          start_masks_f[0].numpy(),
        #                          #start_rgb.numpy(),
        #                          goal_angles.numpy(),
        #                          goal_pts[0].cpu().numpy(),
        #                          goal_poses_f[0].numpy(),
        #                          goal_masks_f[0].numpy(),
        #                          #goal_rgb.numpy(),
        #                          init_pose_loss.data[0],
        #                          init_deg_errors)
        #
        #     # Start saving frames
        #     save_dir = pargs.save_dir + "/frames/test" + str(int(k)) + "/"
        #     util.create_dir(save_dir)  # Create directory
        #     pangolin.start_saving_frames(save_dir)  # Start saving frames
        #     print("Rendering frames for example: {} and saving them to: {}".format(k, save_dir))
        #
        #     for j in xrange(len(curr_angles_s)):
        #         if (j % 10 == 0):
        #             print("Saving frame: {}/{}".format(j, len(curr_angles_s)))
        #         pangolin.update_curr(curr_angles_s[j].numpy(),
        #                              curr_pts_s[j].numpy(),
        #                              curr_poses_s[j].numpy(),
        #                              curr_masks_s[j].numpy(),
        #                              #curr_rgb_s[j].numpy(),
        #                              loss_s[j],
        #                              curr_deg_errors_s[j].numpy(),
        #                              1)  # Save frame
        #
        #     # Stop saving frames
        #     time.sleep(1)
        #     pangolin.stop_saving_frames()

    # # Print all errors
    # i_err, f_err = torch.cat(init_errors,0), torch.cat(final_errors,0)
    # print("------------ INITIAL ERRORS -------------")
    # print(i_err)
    # print("------------- FINAL ERRORS --------------")
    # print(f_err)
    # num_top6_strict = (f_err[:,:6] < 1.5).sum(1).eq(6).sum()
    # num_top4_strict = (f_err[:,:4] < 1.5).sum(1).eq(4).sum()
    # num_top6_relax  = ((f_err[:,:4] < 1.5).sum(1).eq(4) *
    #                    (f_err[:,[4,5]] < 3.0).sum(1).eq(2)).sum()
    # print("------ Top 6 jts strict: {}/{}, relax: {}/{}. Top 4 jts strict: {}/{}".format(
    #     num_top6_strict, num_configs, num_top6_relax, num_configs, num_top4_strict, num_configs
    # ))

    # # Save data stats
    # if pargs.save_frame_stats:
    #     torch.save(datastats, pargs.save_dir + '/datastats.pth.tar')

def integrate_ctrls(start_angles, ctrls, dt, pargs):
    # Do forward integration of the ctrls
    angle_traj = torch.zeros(ctrls.size(0)+1, start_angles.nelement())
    angle_traj[0] = start_angles
    for h in xrange(ctrls.size(0)):
        angle_traj[h+1] = angle_traj[h] + (ctrls[h] * dt)

    # Constrain output traj, not the controls themselves!
    start_angles_v = start_angles.view(1,-1).expand_as(angle_traj)
    if pargs.only_top4_jts:
        angle_traj[:,4:] = start_angles_v[:,4:]
    elif pargs.only_top6_jts:
        angle_traj[:,6:] = start_angles_v[:,6:]

    # Return
    return angle_traj

### Optimizer that takes in state & controls, returns a loss and loss gradient
def optimize_ctrl(ctrls, ctrlsgrad, param, transmodel, start_poses, goal_poses,
                  start_angles, goal_angles, init_traj):
    # ============ FWD pass + Compute loss ============#
    # Roll out over the horizon
    start_pose_v = Variable(start_poses.data.clone(), requires_grad=False)
    goal_pose_v  = Variable(goal_poses.data.clone(), requires_grad=False)
    ctrls_v      = Variable(torch.FloatTensor(ctrls).view(pargs.horizon, -1).type(pargs.deftype), requires_grad=True)
    pred_poses_v = []
    iter_loss = torch.zeros(pargs.horizon)
    pose_loss_v = 0
    for h in xrange(pargs.horizon):
        # Get inputs for the current timestep
        pose = start_pose_v if (h == 0) else pred_poses_v[-1]
        ctrl = ctrls_v.narrow(0, h, 1)

        # Run a forward pass through the network
        _, pred_pose = transmodel([pose, ctrl])
        pred_poses_v.append(pred_pose)

        # Compute loss and add to list of losses
        if h > 0.75 * pargs.horizon:
            loss = pargs.loss_wt * ctrlnets.BiMSELoss(pred_pose, goal_pose_v)
            #loss = pargs.loss_wt * ctrlnets.BiAbsLoss(pred_pose, goal_pose_v) * ((h+1.) / pargs.horizon)
            pose_loss_v += loss
            iter_loss[h] = loss.data[0]

    # Add a control smoothness cost
    ctrl_vel = (ctrls_v[1:] - ctrls_v[:-1]) / (dt * args.step_len)
    ctrl_vel_loss = pargs.smooth_wt * ctrl_vel.pow(2).mean()

    # Add losses together
    curr_loss_v = pose_loss_v + ctrl_vel_loss
    pose_loss, smooth_loss, total_loss = pose_loss_v.data[0], ctrl_vel_loss.data[0], curr_loss_v.data[0]

    # ============ BWD pass ============#

    # Backward pass & optimize
    transmodel.zero_grad()  # Zero gradients
    zero_gradients(ctrls_v)  # Zero gradients for controls
    curr_loss_v.backward()  # Compute gradients - BWD pass
    ctrl_grad = ctrls_v.grad.data.cpu().float()

    # ============ For visualization only ============#
    # Get the controls input and integrate them
    ctrls_t = ctrls_v.data.cpu().clone()
    angle_traj = integrate_ctrls(start_angles, ctrls_t, dt*args.step_len, pargs)

    # Print losses and errors
    pargs.curr_iter += 1
    curr_deg_error = (angle_traj[-1] - goal_angles) * (180.0 / np.pi)
    init_deg_error = (init_traj[-1] - goal_angles) * (180.0 / np.pi)
    total_deg_error = (start_angles - goal_angles) * (180.0 / np.pi)
    print('Iter: {}/{}, Loss: {} ({}/{})'.format(pargs.curr_iter, pargs.max_iter, total_loss, pose_loss, smooth_loss))
    print('Joint angle errors in degrees (curr/init): ',
          torch.cat([curr_deg_error.view(7,1), init_deg_error.view(7,1), total_deg_error.view(7,1)], 1))

    # Request termination if we have reached a max num of iters
    if pargs.curr_iter == pargs.max_iter:
        print("Terminating as optimizer reached max number of function evaluations: {}".format(pargs.curr_iter))
        if pargs.optimizer == 'xalglib_cg':
            xalglib.mincgrequesttermination(pargs.xalglib_optim_state)
        elif pargs.optimizer == 'xalglib_lbfgs':
            xalglib.minlbfgsrequesttermination(pargs.xalglib_optim_state)

    # Return loss and gradients
    for k in xrange(ctrl_grad.view(-1).nelement()):
        ctrlsgrad[k] = ctrl_grad.view(-1)[k]
    return curr_loss_v.data[0]

### Compute a point cloud give the arm config
# Assumes that a "Tensor" is the input, not a "Variable"
def generate_ptcloud(config):
    # Render the config & get the point cloud
    assert(not util.is_var(config))
    config_f = config.view(-1).clone().float()
    pts      = torch.FloatTensor(1, 3, args.img_ht, args.img_wd)
    labels   = torch.FloatTensor(1, 1, args.img_ht, args.img_wd)
    pangolin.render_arm(config_f.numpy(), pts[0].numpy(), labels[0].numpy())
    return pts.type_as(config), labels.type_as(config)

### Compute poses give the arm config
# Assumes that a "Tensor" is the input, not a "Variable"
def generate_poses(config, mesh_ids, model_view, nposes=50):
    # Render the config & get the poses
    assert(not util.is_var(config))
    config_f = config.view(-1).clone().float()
    allposes = torch.FloatTensor(1, nposes, 3, 4).zero_()
    nposes_i = torch.IntTensor(1)
    pangolin.render_pose(config_f.numpy(), allposes[0].numpy(), nposes_i.numpy())
    print(nposes_i)
    
    # Get the correct poses (based on mesh ids)
    num_meshes = mesh_ids.nelement()  # Num meshes
    poses    = torch.FloatTensor(1, num_meshes+1, 3, 4).zero_()
    poses[0,0,:,0:3] = torch.eye(3).float() # Identity transform for BG
    for j in xrange(num_meshes):
        meshid = mesh_ids[j]
        # Transform using modelview matrix
        tfm = torch.eye(4)
        tfm[0:3,:] = allposes[0][meshid][0:3,:]
        se3tfm = torch.mm(model_view, tfm)  # NOTE: Do matrix multiply, not * (cmul) here
        poses[0,j+1] = se3tfm[0:3,:]  # 3 x 4 transform
    return poses.type_as(config)

### Compute masks
def compute_masks_from_labels(labels, mesh_ids):
    masks = torch.FloatTensor(1, mesh_ids.nelement()+1, args.img_ht, args.img_wd).type_as(labels)
    labels.round_() # Round off the labels
    # Compute masks based on the labels and mesh ids (BG is channel 0, and so on)
    # Note, we have saved labels in channel 0 of masks, so we update all other channels first & channel 0 (BG) last
    num_meshes = mesh_ids.nelement()
    for j in xrange(num_meshes):
        masks[:, j+1] = labels.eq(mesh_ids[j])  # Mask out that mesh ID
        if (j == num_meshes - 1):
            masks[:, j+1] = labels.ge(mesh_ids[j])  # Everything in the end-effector
    masks[:, 0] = masks.narrow(1, 1, num_meshes).sum(1).eq(0)  # All other masks are BG
    return masks

################ RUN MAIN
if __name__ == '__main__':
    main()
