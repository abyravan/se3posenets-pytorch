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

##########
# Parse arguments
parser = argparse.ArgumentParser(description='Reactive control using SE3-Pose-Nets')

parser.add_argument('--pose-checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for training (default: none)')
parser.add_argument('--trans-checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for training (default: none)')

# Problem options
parser.add_argument('--start-id', default=-1, type=int, metavar='N',
                    help='ID in the test dataset for the start state (default: -1 = randomly chosen)')
parser.add_argument('--goal-horizon', default=1.5, type=float, metavar='SEC',
                    help='Planning goal horizon in seconds (default: 1.5)')
parser.add_argument('--only-top4-jts', action='store_true', default=False,
                    help='Controlling only the first 4 joints (default: False)')
parser.add_argument('--only-top6-jts', action='store_true', default=False,
                    help='Controlling only the first 6 joints (default: False)')
parser.add_argument('--ctrl-specific-jts', type=str, default='', metavar='JTS',
                    help='Comma separated list of joints to control. All other jts will have 0 error '
                         'but the system can move those (default: '' => all joints are controlled)')

# Planner options
parser.add_argument('--optimization', default='gn', type=str, metavar='OPTIM',
                    help='Type of optimizer to use: [gn] | backprop')
parser.add_argument('--max-iter', default=100, type=int, metavar='N',
                    help='Maximum number of planning iterations (default: 100)')
parser.add_argument('--gn-perturb', default=1e-3, type=float, metavar='EPS',
                    help='Perturbation for the finite-differencing to compute the jacobian (default: 1e-3)')
parser.add_argument('--gn-lambda', default=1e-4, type=float, metavar='LAMBDA',
                    help='Damping constant (default: 1e-4)')
parser.add_argument('--gn-jac-check', action='store_true', default=False,
                    help='check FD jacobian & gradient against the numerical jacobian & backprop gradient (default: False)')
parser.add_argument('--max-ctrl-mag', default=1.0, type=float, metavar='UMAX',
                    help='Maximum allowable control magnitude (default: 1 rad/s)')
parser.add_argument('--ctrl-mag-decay', default=0.99, type=float, metavar='W',
                    help='Decay the control magnitude by scaling by this weight after each iter (default: 0.99)')
parser.add_argument('--loss-scale', default=1000, type=float, metavar='WT',
                    help='Scaling factor for the loss (default: 1000)')
parser.add_argument('--loss-threshold', default=0, type=float, metavar='EPS',
                    help='Threshold for convergence check based on the losses (default: 0)')

# TODO: Add criteria for convergence

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
parser.add_argument('--save-frames', action='store_true', default=False,
                    help='Enables post-saving of generated frames, very slow process (default: False)')
parser.add_argument('--save-frame-stats', action='store_true', default=False,
                    help='Saves all necessary data for genering rendered frames later (default: False)')

def main():
    # Parse args
    global pargs, args, num_train_iter
    pargs = parser.parse_args()
    pargs.cuda = not pargs.no_cuda and torch.cuda.is_available()

    # Create save directory and start tensorboard logger
    if pargs.save_dir == '':
        checkpoint_dir = pargs.pose_checkpoint.rpartition('/')[0]
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
    if pargs.cuda:
        torch.cuda.manual_seed(pargs.seed)

    # Default tensor type
    deftype = 'torch.cuda.FloatTensor' if pargs.cuda else 'torch.FloatTensor' # Default tensor type

    # Invert a matrix to initialize the torch inverse code
    temp = torch.rand(7,7).type(deftype)
    tempinv = torch.inverse(temp)

    ########################
    ############ Load pre-trained network

    # Load data from checkpoint
    # TODO: Print some stats on the training so far, reset best validation loss, best epoch etc
    if os.path.isfile(pargs.pose_checkpoint) and os.path.isfile(pargs.trans_checkpoint):
        print("=> loading checkpoints '{}, {}'".format(pargs.pose_checkpoint, pargs.trans_checkpoint))
        pose_checkpoint   = torch.load(pargs.pose_checkpoint)
        trans_checkpoint  = torch.load(pargs.trans_checkpoint)
        args              = pose_checkpoint['args']
        try:
            num_train_iter = pose_checkpoint['num_train_iter']
        except:
            num_train_iter = pose_checkpoint['epoch'] * args.train_ipe
        print("=> loaded checkpoint (pose epoch: {}, num train iter: {}, trans epoch: {})"
              .format(pose_checkpoint['epoch'], num_train_iter, trans_checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}, {}'".format(pargs.pose_checkpoint, pargs.trans_checkpoint))
        raise RuntimeError

    # BWDs compatibility
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

    # # Create a model
    # if args.seq_len == 1:
    #     if args.use_gt_masks:
    #         model = ctrlnets.SE3OnlyPoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
    #                                           se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
    #                                           input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
    #                                           init_posese3_iden=False, init_transse3_iden=False,
    #                                           use_wt_sharpening=args.use_wt_sharpening,
    #                                           sharpen_start_iter=args.sharpen_start_iter,
    #                                           sharpen_rate=args.sharpen_rate, pre_conv=False,
    #                                           wide=args.wide_model, use_jt_angles=args.use_jt_angles,
    #                                           use_jt_angles_trans=args.use_jt_angles_trans,
    #                                           num_state=args.num_state_net)  # TODO: pre-conv
    #         posemaskpredfn = model.posemodel.forward
    #     elif args.use_gt_poses:
    #         assert False, "No need to run tests with GT poses provided"
    #     else:
    #         model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
    #                                       se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
    #                                       input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
    #                                       init_posese3_iden=False, init_transse3_iden=False,
    #                                       use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
    #                                       sharpen_rate=args.sharpen_rate, pre_conv=False, wide=args.wide_model,
    #                                       use_jt_angles=args.use_jt_angles, use_jt_angles_trans=args.use_jt_angles_trans,
    #                                       num_state=args.num_state_net) # TODO: pre-conv
    #         posemaskpredfn = model.posemaskmodel.forward
    # else:
    #     if args.use_gt_masks:
    #         modelfn = ctrlnets.MultiStepSE3OnlyPoseModel
    #     elif args.use_gt_poses:
    #         assert False, "No need to run tests with GT poses provided"
    #     else:
    modelfn = ctrlnets.MultiStepSE3NoTransModel
    posemodel = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3, delta_pivot=args.delta_pivot,
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
    posemaskpredfn = posemodel.forward_only_pose if args.use_gt_masks else posemodel.forward_pose_mask
    if pargs.cuda:
        posemodel.cuda() # Convert to CUDA if enabled

    # Update parameters from trained network
    try:
        posemodel.load_state_dict(pose_checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        posemodel.load_state_dict(pose_checkpoint['model_state_dict'])

    # Set model to evaluate mode
    posemodel.eval()

    #### Trans model
    transmodel = ctrlnets.TransitionModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3, delta_pivot=args.delta_pivot,
                                          se3_type=args.se3_type, use_kinchain=False,
                                          nonlinearity=args.nonlinearity, init_se3_iden=args.init_transse3_iden,
                                          local_delta_se3=args.local_delta_se3,
                                          use_jt_angles=args.use_jt_angles_trans, num_state=args.num_state_net)
    if pargs.cuda:
        transmodel.cuda() # Convert to CUDA if enabled

    # Update parameters from trained network
    try:
        transmodel.load_state_dict(trans_checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        transmodel.load_state_dict(trans_checkpoint['model_state_dict'])

    # Set model to evaluate mode
    transmodel.eval()

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

    # Get dimensions of ctrl & state
    try:
        statelabels, ctrllabels = data.read_statectrllabels_file(data_path + "/statectrllabels.txt")
        print("Reading state/ctrl joint labels from: " + data_path + "/statectrllabels.txt")
    except:
        statelabels = data.read_statelabels_file(data_path + '/statelabels.txt')['frames']
        ctrllabels = statelabels  # Just use the labels
        print("Could not read statectrllabels file. Reverting to labels in statelabels file")
    args.num_state, args.num_ctrl = len(statelabels), len(ctrllabels)
    print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))

    # Find the IDs of the controlled joints in the state vector
    # We need this if we have state dimension > ctrl dimension and
    # if we need to choose the vals in the state vector for the control
    ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
    print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

    '''
    # Get cam data
    args.cam_extrinsics = data.read_cameradata_file(data_path + '/cameradata.txt') # TODO: BWDs compatibility
    args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               args.cam_intrinsics) # TODO: BWDs compatibility
    baxter_data = data.read_recurrent_baxter_dataset(data_path, args.img_suffix,
                                                     step_len=1, seq_len=1,
                                                     train_per=0.6, val_per=0.15)
    disk_read_func = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht=args.img_ht, img_wd=args.img_wd,
                                                                      img_scale=args.img_scale,
                                                                      ctrl_type='actdiffvel',
                                                                      num_ctrl=args.num_ctrl, num_state=args.num_state,
                                                                      mesh_ids=args.mesh_ids, ctrl_ids=ctrlids_in_state,
                                                                      camera_extrinsics=args.cam_extrinsics,
                                                                      camera_intrinsics=args.cam_intrinsics)
    filter_func = lambda b: data.filter_func(b, mean_dt=args.mean_dt, std_dt=args.std_dt)
    test_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test', filter_func)  # Test dataset

    # Get start & goal samples
    start_id = pargs.start_id if (pargs.start_id >= 0) else np.random.randint(len(test_dataset))
    goal_id  = start_id + round(pargs.goal_horizon/dt)
    print('Test dataset size: {}, Start ID: {}, Goal ID: {}, Duration: {}'.format(len(test_dataset),
                                  start_id, goal_id, pargs.goal_horizon))
    start_sample = test_dataset[start_id]
    goal_sample  = test_dataset[goal_id]
    
    # Get the joint angles
    start_angles = start_sample['actconfigs'][0]
    goal_angles  = goal_sample['actconfigs'][0]
    '''

    # #########
    # # SOME TEST CONFIGS:
    # # TODO: Get more challenging test configs by moving arm physically
    start_angles_all = torch.FloatTensor(
        [
         [6.5207e-01, -2.6608e-01,  8.2490e-01,  1.0400e+00, -2.9203e-01,  2.1293e-01, 1.1197e-06],
         [0.1800, 0.4698, 1.5043, 0.5696, 0.1862, 0.8182, 0.0126],
         [1.0549, -0.1554, 1.2620, 1.0577, 1.0449, -1.2097, -0.6803],
         [-1.2341e-01, 7.4693e-01, 1.4739e+00, 1.6523e+00, -2.6991e-01, 1.1523e-02, -9.5822e-05],
         [-0.5728, 0.6794, 1.4149, 1.7189, -0.6503, 0.3657, -1.6146],
         [0.5426, 0.4880, 1.4143, 1.2573, -0.4632, -1.0516, 0.1703],
         [0.4412, -0.5150, 0.8153, 1.5142, 0.4762, 0.0438, 0.7105],
         [0.0619, 0.1619, 1.1609, 0.9808, 0.3923, 0.6253, 0.0328],
         [0.5052, -0.4135, 1.0945, 1.6024, 1.0821, -0.6957, -0.2535],
         [-0.6730, 0.5814, 1.3403, 1.7309, -0.4106, 0.4301, -1.7868],
         [0.3525, -0.1269, 1.1656, 1.3804, 0.1220, 0.3742, -0.1250]
        ]
        )
    goal_angles_all  = torch.FloatTensor(
        [
         [0.5793, -0.0749,  0.9222,  1.4660, -0.7369, -0.6797, -0.4747],
         [0.1206, 0.2163, 1.2128, 0.9753, 0.2447, 0.5462, -0.2298],
         [0.0411, -0.4383, 1.1090, 1.9053, 0.7874, -0.1648, -0.3210],
         [8.3634e-01, -3.7185e-01, 5.8938e-01, 1.0404e+00, 3.0321e-01, 1.6204e-01, -1.9278e-05],
         [-0.5702, 0.6332, 1.4110, 1.6701, -0.5085, 0.4071, -1.6792],
         [0.0338, 0.0829, 1.0422, 1.6009, -0.7885, -0.5373, 0.1593],
         [0.2692, -0.4469, 0.6287, 0.8841, 0.2070, 1.3161, 0.4913],
         [4.1391e-01, -4.5127e-01, 8.9605e-01, 1.1968e+00, -4.4754e-05, 8.8374e-01, 6.2656e-02],
         [0.0880, -0.3266, 0.8092, 1.1611, 0.2845, 0.5481, -0.4666],
         [-0.0374, -0.2891, 1.2771, 1.4422, -0.4017, 0.9142, -0.7823],
         [0.4959, -0.2184, 1.2100, 1.8197, 0.3975, -0.7801, 0.2076]
        ]
        )

    # ### More test configs
    # start_angles_all = torch.FloatTensor(
    #     [
    #         [-0.12341, 0.74693, 1.4739, 1.6523, -0.26991, 0.011523, -0.0009],
    #         [0.0619, 0.1619, 1.1609, 0.9808, 0.3923, 0.6253, 0.0328],
    #         [1.0549, -0.1554, 1.2620, 1.0577, 1.0449, -1.2097, -0.6803],
    #     ]
    # )
    #
    # goal_angles_all = torch.FloatTensor(
    #     [
    #         [0.83634, -0.37185, 0.58938, 1.0404, 0.50321, 0.67204, 0.0002],
    #         [0.8139, -0.6512, 0.596, 1.5968, -4.4754e-05, -1.25, 6.2656e-02],
    #         [0.0411, -0.8383, 0.590, 1.9053, 0.1874, -0.1648, -0.3210],
    #     ]
    # )

    # # ###
    # start_angles_all = torch.FloatTensor(
    #      [
    #          [0.0619, 0.1619, 1.1609, 0.9808, 0.3923, 0.6253, 0.0328],
    #          [-0.12341, 0.74693, 1.4739, 1.6523, -0.26991, 0.011523, -0.0009],
    #          [1.0549, -0.1554, 1.2620, 1.0577, 1.0449, -1.2097, -0.6803],
    #      ]
    # )
    #
    # goal_angles_all = torch.FloatTensor(
    #     [
    #         [0.8139, -0.6512, 0.596, 1.5968, -4.4754e-05, -1.25, 6.2656e-02],
    #         [0.83634, -0.37185, 0.58938, 1.0404, 0.50321, 0.67204, 0.0002],
    #         [0.0411, -0.8383, 0.590, 1.9053, 0.1874, -0.1648, -0.3210],
    #     ]
    # )

    # Iterate over test configs
    num_configs = start_angles_all.size(0)
    if pargs.num_configs > 0:
        num_configs = min(start_angles_all.size(0), pargs.num_configs)
    print("Running tests over {} configs".format(num_configs))
    iterstats = []
    init_errors, final_errors = [], []
    datastats = []
    for k in xrange(num_configs):
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
            for k in xrange(7):
                if k not in ctrl_jts:
                    goal_angles[k] = start_angles[k]

        ########################
        ############ Get start & goal point clouds, predict poses & masks
        # Initialize problem
        #start_pts, da_goal_pts = torch.zeros(1,3,args.img_ht,args.img_wd), torch.zeros(1,3,args.img_ht,args.img_wd)
        #pangolin.init_problem(start_angles.numpy(), goal_angles.numpy(), start_pts[0].numpy(), da_goal_pts[0].numpy())

        # Get full goal point cloud
        start_pts, _ = generate_ptcloud(start_angles)
        goal_pts, _  = generate_ptcloud(goal_angles)

        #### Predict start/goal poses and masks
        print('Predicting start/goal poses and masks')
        #if args.use_jt_angles or args.seq_len > 1:
        sinp = [util.to_var(start_pts.type(deftype)), util.to_var(start_angles.view(1, -1).type(deftype))]
        tinp = [util.to_var(goal_pts.type(deftype)), util.to_var(goal_angles.view(1, -1).type(deftype))]
        #else:
        #    sinp = util.to_var(start_pts.type(deftype))
        #    tinp = util.to_var(goal_pts.type(deftype))

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

        # Render the poses
        # NOTE: Data passed into cpp library needs to be assigned to specific vars, not created on the fly (else gc will free it)
        #start_poses_f, goal_poses_f = start_poses.data.cpu().float(), goal_poses.data.cpu().float()
        #pangolin.initialize_poses(start_poses_f[0].numpy(), goal_poses_f[0].numpy())

        # Print error
        print('Initial jt angle error:')
        full_deg_error = (start_angles-goal_angles) * (180.0/np.pi) # Full error in degrees
        print(full_deg_error.view(7,1))
        init_errors.append(full_deg_error.view(1,7))

        # Compute initial pose loss
        init_pose_loss = args.loss_scale * ctrlnets.BiMSELoss(start_poses, goal_poses)  # Get distance from goal
        init_pose_err_indiv = args.loss_scale * (start_poses - goal_poses).pow(2).view(args.num_se3,12).mean(1)
        init_deg_errors = full_deg_error.view(7).numpy()

        # Render the poses
        # NOTE: Data passed into cpp library needs to be assigned to specific vars, not created on the fly (else gc will free it)
        print("Initializing the visualizer...")
        start_masks_f, goal_masks_f = start_masks.data.cpu().float(), goal_masks.data.cpu().float()
        start_poses_f, goal_poses_f = start_poses.data.cpu().float(), goal_poses.data.cpu().float()
        #start_rgb = torch.zeros(3, args.img_ht, args.img_wd).byte()
        #goal_rgb = torch.zeros(3, args.img_ht, args.img_wd).byte()
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

        # For saving:
        if pargs.save_frame_stats:
            initstats = [start_angles.numpy(),
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
                         init_deg_errors]

        ########################
        ############ Run the controller
        # Init stuff
        ctrl_mag = pargs.max_ctrl_mag
        angles, deg_errors = [start_angles], [full_deg_error.view(1,7)]
        ctrl_grads, ctrls  = [], []
        losses = []

        # Init vars for all items
        init_ctrl_v  = util.to_var(torch.zeros(1,args.num_ctrl).type(deftype), requires_grad=True) # Need grad w.r.t this
        goal_poses_v = util.to_var(goal_poses.data, requires_grad=False)

        # # Plots for errors and loss
        # fig, axes = plt.subplots(2, 1)
        # fig.show()

        # Save for final frame saving
        if pargs.save_frames or pargs.save_frame_stats:
            curr_angles_s, curr_pts_s, curr_poses_s, curr_masks_s = [], [], [] ,[]
            #curr_rgb_s = []
            loss_s, err_indiv_s, curr_deg_errors_s = [], [], []

        # Run the controller for max_iter iterations
        gen_time, posemask_time, optim_time, viz_time, rest_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        conv_iter = pargs.max_iter
        inc_ctr, max_ctr = 0, 10 # Num consecutive times we've seen an increase in loss
        status, prev_loss = 0, np.float("inf")
        for it in xrange(pargs.max_iter):
            # Print
            print('\n #####################')

            # Get current point cloud
            start = time.time()
            curr_angles = angles[it]
            curr_pts, curr_rlabels = generate_ptcloud(curr_angles)
            curr_pts = curr_pts.type(deftype)
            gen_time.update(time.time() - start)

            # Predict poses and masks
            start = time.time()
            #if args.use_jt_angles or args.seq_len > 1:
            inp = [util.to_var(curr_pts), util.to_var(curr_angles.view(1, -1).type(deftype))]
            #else:
            #    inp = util.to_var(curr_pts)
            if args.use_gt_masks:
                curr_masks = util.to_var(compute_masks_from_labels(curr_rlabels, args.mesh_ids).type(deftype))
                curr_poses = posemaskpredfn(inp)
            else:
                curr_poses, curr_masks = posemaskpredfn(inp, train_iter=num_train_iter)

            # Update poses (based on pivots)
            curr_poses, _, _ = ctrlnets.update_pose_centers(util.to_var(curr_pts), curr_masks, curr_poses, args.pose_center)

            # Get CPU stuff
            curr_poses_f, curr_masks_f = curr_poses.data.cpu().float(), curr_masks.data.cpu().float()
            #curr_rgb = start_rgb
            posemask_time.update(time.time() - start)

            # Compute pivots
            curr_pivots = None
            if not ((args.delta_pivot == '') or (args.delta_pivot == 'pred')):
                curr_pivots = ctrlnets.compute_pivots(util.to_var(curr_pts), curr_masks, curr_poses, args.delta_pivot)

            # Render poses and masks using Pangolin
            start = time.time()
            #_, curr_labels = curr_masks_f.max(dim=1)
            #curr_labels_f = curr_labels.float()
            #pangolin.update_masklabels_and_poses(curr_labels_f.numpy(), curr_poses_f[0].numpy())

            # Show masks using tensor flow
            if (it % args.disp_freq) == 0:
                maskdisp = torchvision.utils.make_grid(curr_masks.data.cpu().view(-1, 1, args.img_ht, args.img_wd),
                                                           nrow=args.num_se3, normalize=True, range=(0, 1))
                info = {'curr masks': util.to_np(maskdisp.narrow(0, 0, 1))}
                for tag, images in info.items():
                    tblogger.image_summary(tag, images, it)
            viz_time.update(time.time() - start)

            # Run one step of the optimization (controls are always zeros, poses change)
            start = time.time()
            ctrl_grad, loss = optimize_ctrl(model=transmodel,
                                            poses=curr_poses, ctrl=init_ctrl_v,
                                            angles=angles[it].view(1,-1),
                                            goal_poses=goal_poses_v,
                                            pivots=curr_pivots)
            optim_time.update(time.time() - start)
            ctrl_grads.append(ctrl_grad.cpu().float()) # Save this

            # Set last 3 joint's controls to zero
            if pargs.only_top4_jts:
                ctrl_grad[4:] = 0
            elif pargs.only_top6_jts:
                ctrl_grad[6:] = 0

            # Decay controls if loss is small
            if loss < 1:
                pargs.ctrl_mag_decay = 0.975
            else:
                ctrl_mag = pargs.max_ctrl_mag
                pargs.ctrl_mag_decay = 1.

            # Get the control direction and scale it by max control magnitude
            start = time.time()
            if ctrl_mag > 0:
                ctrl_dirn = ctrl_grad.cpu().float() / ctrl_grad.norm(2) # Dirn
                curr_ctrl = ctrl_dirn * ctrl_mag # Scale dirn by mag
                ctrl_mag *= pargs.ctrl_mag_decay # Decay control magnitude
            else:
                curr_ctrl = ctrl_grad.cpu().float()

            # Apply control (simple velocity integration)
            next_angles = curr_angles - (curr_ctrl * dt)

            # Save stuff
            losses.append(loss)
            ctrls.append(curr_ctrl)
            angles.append(next_angles)
            deg_errors.append((next_angles-goal_angles).view(1,7)*(180.0/np.pi))

            # Print losses and errors
            print('Test: {}/{}, Ctr: {}/{}, Control Iter: {}/{}, Loss: {}'.format(k+1, num_configs, inc_ctr, max_ctr,
                                                                                  it+1, pargs.max_iter, loss))
            print('Joint angle errors in degrees: ',
                  torch.cat([deg_errors[-1].view(7,1), full_deg_error.unsqueeze(1)], 1))

            #### Render stuff
            curr_pts_f = curr_pts.cpu().float()
            curr_deg_errors = (next_angles - goal_angles).view(1, 7) * (180.0 / np.pi)
            curr_pose_err_indiv = args.loss_scale * (curr_poses - goal_poses).pow(2).view(args.num_se3, 12).mean(1)
            pangolin.update_curr(curr_angles.numpy(),
                                 curr_pts_f[0].numpy(),
                                 curr_poses_f[0].numpy(),
                                 curr_masks_f[0].numpy(),
                                 #curr_rgb.numpy(),
                                 loss,
                                 curr_pose_err_indiv.data.cpu().numpy(),
                                 curr_deg_errors[0].numpy(),
                                 0)  # Don't save frame

            # Save for future frame generation!
            if pargs.save_frames or pargs.save_frame_stats:
                curr_angles_s.append(curr_angles)
                curr_pts_s.append(curr_pts_f[0])
                curr_poses_s.append(curr_poses_f[0])
                curr_masks_s.append(curr_masks_f[0])
                #curr_rgb_s.append(curr_rgb)
                loss_s.append(loss)
                err_indiv_s.append(curr_pose_err_indiv.data.cpu())
                curr_deg_errors_s.append(curr_deg_errors[0])

            # # Plot the errors & loss
            # colors = ['r', 'g', 'b', 'c', 'y', 'k', 'm']
            # labels = []
            # if ((it % 4) == 0) or (loss < pargs.loss_threshold):
            #     conv = "Converged" if (loss < pargs.loss_threshold) else ""
            #     axes[0].set_title(("Test: {}/{}, Iter: {}, Loss: {}, Jt angle errors".format(k+1, num_configs,
            #                                                                                  (it + 1), loss)) + conv)
            #     for j in xrange(7):
            #         axes[0].plot(torch.cat(deg_errors, 0).numpy()[:, j], color=colors[j])
            #         labels.append("Jt-{}".format(j))
            #     # I'm basically just demonstrating several different legend options here...
            #     axes[0].legend(labels, ncol=4, loc='upper center',
            #                    bbox_to_anchor=[0.5, 1.1],
            #                    columnspacing=1.0, labelspacing=0.0,
            #                    handletextpad=0.0, handlelength=1.5,
            #                    fancybox=True, shadow=True)
            #     axes[1].set_title("Iter: {}, Loss".format(it + 1))
            #     axes[1].plot(losses, color='k')
            #     fig.canvas.draw()  # Render
            #     plt.pause(0.01)
            # if (it % args.disp_freq) == 0: # Clear now and then
            #     for ax in axes:
            #         ax.cla()

            # Finish
            rest_time.update(time.time() - start)
            print('Gen: {:.3f}({:.3f}), PoseMask: {:.3f}({:.3f}), Viz: {:.3f}({:.3f}),'
                  ' Optim: {:.3f}({:.3f}), Rest: {:.3f}({:.3f})'.format(
                gen_time.val, gen_time.avg, posemask_time.val, posemask_time.avg,
                viz_time.val, viz_time.avg, optim_time.val, optim_time.avg,
                rest_time.val, rest_time.avg))

            # Check for convergence in pose space
            if loss < pargs.loss_threshold:
                print("*****************************************")
                print('Control Iter: {}/{}, Loss: {} less than threshold'.format(it + 1, pargs.max_iter,
                                                                                 loss, pargs.loss_threshold))
                conv_iter = it+1
                print("*****************************************")
                break

            # Check loss increase
            if (loss > prev_loss):
                inc_ctr += 1
                if inc_ctr == max_ctr:
                    print("************* FAILED *****************")
                    print('Control Iter: {}/{}, Loss: {} increased by more than 5 times in a window of the last 10 states'.format(it + 1, pargs.max_iter,
                                                                                     loss, pargs.loss_threshold))
                    status = -1
                    conv_iter = it + 1
                    print("*****************************************")
                    break
            else:
                inc_ctr = max(inc_ctr-1, 0) # Reset
            prev_loss = loss

        # Print final stats
        print('=========== FINISHED ============')
        print('Final loss after {} iterations: {}'.format(pargs.max_iter, losses[-1]))
        print('Final angle errors in degrees: ')
        print(deg_errors[-1].view(7,1))
        final_errors.append(deg_errors[-1])

        # Print final error in stats file
        if status == -1:
            conv = "Final Loss: {}, Failed after {} iterations \n".format(losses[-1], conv_iter)
        elif conv_iter == pargs.max_iter:
            conv = "Final Loss: {}, Did not converge after {} iterations\n".format(losses[-1], pargs.max_iter)
        else:
            conv = "Final Loss: {}, Converged after {} iterations\n".format(losses[-1], conv_iter)
        errorfile.write(("Test: {}, ".format(k+1)) + conv)
        for j in xrange(len(start_angles)):
            errorfile.write("{}, {}, {}, {}\n".format(start_angles[j], goal_angles[j],
                                                    full_deg_error[j], deg_errors[-1][0,j]))
        errorfile.write("\n")

        # Save stats and exit
        iterstats.append({'start_angles': start_angles, 'goal_angles': goal_angles,
                          'angles': angles, 'ctrls': ctrls,
                          'predctrls': ctrl_grads,
                          'deg_errors': deg_errors, 'losses': losses, 'status': status})

        # Save
        if pargs.save_frame_stats:
            datastats.append([initstats, curr_angles_s, curr_pts_s, curr_poses_s,
                              curr_masks_s, loss_s, err_indiv_s, curr_deg_errors_s]) # curr_rgb_s

        ###################### RE-RUN VIZ TO SAVE FRAMES TO DISK CORRECTLY
        ######## Saving frames to disk now!
        if pargs.save_frames:
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
                                 init_deg_errors)

            # Start saving frames
            save_dir = pargs.save_dir + "/frames/test" + str(int(k)) + "/"
            util.create_dir(save_dir)  # Create directory
            pangolin.start_saving_frames(save_dir)  # Start saving frames
            print("Rendering frames for example: {} and saving them to: {}".format(k, save_dir))

            for j in xrange(len(curr_angles_s)):
                if (j % 10 == 0):
                    print("Saving frame: {}/{}".format(j, len(curr_angles_s)))
                pangolin.update_curr(curr_angles_s[j].numpy(),
                                     curr_pts_s[j].numpy(),
                                     curr_poses_s[j].numpy(),
                                     curr_masks_s[j].numpy(),
                                     #curr_rgb_s[j].numpy(),
                                     loss_s[j],
                                     curr_deg_errors_s[j].numpy(),
                                     1)  # Save frame

            # Stop saving frames
            time.sleep(1)
            pangolin.stop_saving_frames()

    # Print all errors
    i_err, f_err = torch.cat(init_errors,0), torch.cat(final_errors,0)
    print("------------ INITIAL ERRORS -------------")
    print(i_err)
    print("------------- FINAL ERRORS --------------")
    print(f_err)
    num_top6_strict = (f_err[:,:6] < 1.5).sum(1).eq(6).sum()
    num_top4_strict = (f_err[:,:4] < 1.5).sum(1).eq(4).sum()
    num_top6_relax  = ((f_err[:,:4] < 1.5).sum(1).eq(4) *
                       (f_err[:,[4,5]] < 3.0).sum(1).eq(2)).sum()
    print("------ Top 6 jts strict: {}/{}, relax: {}/{}. Top 4 jts strict: {}/{}".format(
        num_top6_strict, num_configs, num_top6_relax, num_configs, num_top4_strict, num_configs
    ))

    # Save data stats
    if pargs.save_frame_stats:
        torch.save(datastats, pargs.save_dir + '/datastats.pth.tar')

    # Save stats across all iterations
    stats = {'args': args, 'pargs': pargs, 'data_path': data_path, 'start_angles_all': start_angles_all,
             'goal_angles_all': goal_angles_all, 'iterstats': iterstats, 'finalerrors': final_errors,
             'num_top6_strict': num_top6_strict, 'num_top4_strict': num_top4_strict,
             'num_top6_relax': num_top6_relax}
    torch.save(stats, pargs.save_dir + '/planstats.pth.tar')
    sys.stdout = backup
    logfile.close()
    errorfile.close()

    # TODO: Save errors to file for easy reading??

### Function to generate the optimized control
# Note: assumes that it get Variables
def optimize_ctrl(model, poses, ctrl, angles, goal_poses, pivots=None):

    # Do specific optimization based on the type
    if pargs.optimization == 'backprop':
        # Model has to be in training mode
        model.train()

        # ============ FWD pass + Compute loss ============#

        # Setup inputs for FWD pass
        poses_1 = util.to_var(poses.data, requires_grad=False)
        ctrl_1  = util.to_var(ctrl.data, requires_grad=True)
        if args.use_jt_angles_trans:
            angles_1 = util.to_var(angles.type_as(poses.data), requires_grad=False)
            inp = [poses_1, angles_1, ctrl_1]
        else:
            inp = [poses_1, ctrl_1]
        if pivots is not None:
            pivots_1 = util.to_var(pivots.data, requires_grad=False)
            inp.append(pivots_1)

        # Run FWD pass & Compute loss
        _, pred_poses = model(inp)
        loss = args.loss_scale * ctrlnets.BiMSELoss(pred_poses, goal_poses) # Get distance from goal

        # ============ BWD pass ============#

        # Backward pass & optimize
        model.zero_grad()  # Zero gradients
        zero_gradients(ctrl_1) # Zero gradients for controls
        loss.backward()  # Compute gradients - BWD pass

        # Return
        return ctrl_1.grad.data.cpu().view(-1).clone(), loss.data[0]
    else:
        # No backprops here
        model.eval()

        # ============ Compute finite differenced controls ============#

        # Setup stuff for perturbation
        eps      = pargs.gn_perturb
        nperturb = args.num_ctrl
        I = torch.eye(nperturb).type_as(ctrl.data)

        # Do perturbation
        poses_p = util.to_var(poses.data.repeat(nperturb+1,1,1,1))              # Replicate poses
        ctrl_p  = util.to_var(ctrl.data.repeat(nperturb+1,1))                   # Replicate controls
        ctrl_p.data[1:, :] += I * eps    # Perturb the controls

        # ============ FWD pass ============#

        # Setup inputs for FWD pass
        if args.use_jt_angles_trans:
            angles_p = util.to_var(angles.repeat(nperturb+1,1).type_as(poses.data))  # Replicate angles
            inp = [poses_p, angles_p, ctrl_p]
        else:
            inp = [poses_p, ctrl_p]
        if pivots is not None:
            pivots_p = util.to_var(pivots.data.repeat(nperturb+1,1,1).type_as(poses.data))  # Replicate pivots
            inp.append(pivots_p)
        #_, pred_poses_p = model([poses_p, ctrl_p])

        # Run FWD pass
        _, pred_poses_p = model(inp)

        # Backprop only over the loss!
        pred_poses = util.to_var(pred_poses_p.data.narrow(0,0,1), requires_grad=True) # Need grad of loss w.r.t true pred
        loss       = args.loss_scale * ctrlnets.BiMSELoss(pred_poses, goal_poses)
        loss.backward()

        # ============ Compute Jacobian & GN-gradient ============#

        # Compute Jacobian
        Jt  = pred_poses_p.data[1:].view(nperturb, -1).clone() # nperturb x posedim
        Jt -= pred_poses_p.data.narrow(0,0,1).view(1, -1).expand_as(Jt) # [ f(x+eps) - f(x) ]
        Jt.div_(eps) # [ f(x+eps) - f(x) ] / eps

        ### Option 1: Compute GN-gradient using torch stuff by adding eps * I
        # This is incredibly slow at the first iteration
        Jinv = torch.inverse(torch.mm(Jt, Jt.t()) + pargs.gn_lambda * I) # (J^t * J + \lambda I)^-1
        ctrl_grad = torch.mm(Jinv, torch.mm(Jt, pred_poses.grad.data.view(-1,1))) # (J^t*J + \lambda I)^-1 * (Jt * g)

        '''
        ### Option 2: Compute GN-gradient using numpy PINV (instead of adding eps * I)
        # Fastest, but doesn't do well on overall planning if we allow controlling all joints
        # If only controlling the top 4 jts this works just as well as the one above.
        Jtn = util.to_np(Jt)
        ctrl_gradn = np.dot(np.linalg.pinv(Jtn, rcond=pargs.gn_lambda).transpose(), util.to_np(pred_poses.grad.data.view(-1,1)))
        ctrl_grad  = torch.from_numpy(ctrl_gradn)
        '''

        '''
        ### Option 3: Compute GN-gradient using numpy INV (add eps * I)
        # Slower than torch
        Jtn, In = util.to_np(Jt), util.to_np(I)
        Jinv = np.linalg.inv(np.dot(Jtn, Jtn.transpose()) + pargs.gn_lambda * In) # (J^t * J + \lambda I)^-1
        ctrl_gradn = np.dot(Jinv, np.dot(Jtn, util.to_np(pred_poses.grad.data.view(-1,1))))
        ctrl_grad  = torch.from_numpy(ctrl_gradn)
        '''

        # ============ Sanity Check stuff ============#
        # Check gradient / jacobian
        if pargs.gn_jac_check:
            # Set model in training mode
            model.train()

            # FWD pass
            poses_1 = util.to_var(poses.data, requires_grad=False)
            ctrl_1 = util.to_var(ctrl.data, requires_grad=True)
            _, pred_poses_1 = model([poses_1, ctrl_1])
            pred_poses_1_v = pred_poses_1.view(1,-1) # View it nicely

            ###
            # Compute Jacobian via multiple backward passes (SLOW!)
            Jt_1 = compute_jacobian(ctrl_1, pred_poses_1_v)
            diff_j = Jt.t() - Jt_1
            print('Jac diff => Min: {}, Max: {}, Mean: {}'.format(diff_j.min(), diff_j.max(), diff_j.abs().mean()))

            ###
            # Compute gradient via single backward pass + loss
            loss = args.loss_scale * ctrlnets.BiMSELoss(pred_poses_1, goal_poses)  # Get distance from goal
            model.zero_grad()  # Zero gradients
            zero_gradients(ctrl_1)  # Zero gradients for controls
            loss.backward()  # Compute gradients - BWD pass
            diff_g = ctrl_1.grad.data - torch.mm(Jt, pred_poses.grad.data.view(-1,1)) # Error between backprop & J^T g from FD
            print('Grad diff => Min: {}, Max: {}, Mean: {}'.format(diff_g.min(), diff_g.max(), diff_g.abs().mean()))

        # Return the Gauss-Newton gradient
        return ctrl_grad.cpu().view(-1).clone(), loss.data[0]

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

### Compute numerical jacobian via multiple back-props
def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    num_outputs = output.size()[1]

    jacobian = torch.zeros(num_outputs, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_outputs):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_variables=True)
        jacobian[i] = inputs.grad.data

    return jacobian

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
