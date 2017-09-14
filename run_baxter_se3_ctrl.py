#!/usr/bin/env python

# ROS imports
import rospy
from dynamic_reconfigure.server import Server
from std_msgs.msg import Empty
import baxter_interface
from baxter_interface import CHECK_VERSION
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

############# Import pangolin
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

# Load pangolin visualizer library
from torchviz import pangoviz
pangolin = pangoviz.PyPangolinViz(seq_len, img_ht, img_wd, img_scale, num_se3,
                                  cam_intrinsics['fx'], cam_intrinsics['fy'],
                                  cam_intrinsics['cx'], cam_intrinsics['cy'],
                                  dt, oldgrippermodel, savedir)

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
import cv2

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import util
from util import AverageMeter

############ Baxter controller
## Joint position control in position mode
class SE3ControlPositionMode(object):
    """

    @param limb: limb on which to run joint control in position mode

    """
    def __init__(self, limb):

        # control parameters
        self._rate = 100.0  # Hz
        self._missed_cmds = 5.0  # Missed cycles before triggering timeout

        # create our limb instance
        self._limb = baxter_interface.Limb(limb)

        self._start_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

    def move_to_pos(self, joint_angles):
        """
        Moves the limb to joint_angles.
        """
        self._limb.move_to_joint_positions(joint_angles)

    def cur_pos(self):
        return self._limb.joint_angles()

    def control_loop(self):
        # given start and end configuration this should loop over
        # optimizing controls with the se3net, updating the se3net
        # with current joint angles and point cloud and moving to new
        # joint angles
        return

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

############ Depth image subscriber
class DepthImageSubscriber:
    def __init__(self, ht, wd, scale, intrinsics):
        self.subscriber = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.callback)
        self.bridge     = CvBridge()
        self.ht, self.wd, self.scale = ht, wd, scale
        self.intrinsics = intrinsics
        self.imgf = None

    def callback(self,data):
        try:
            self.imgf = self.bridge.imgmsg_to_cv2(data, "16UC1").astype(np.int16) * self.scale
        except CvBridgeError as e:
            print(e)

    def get_ptcloud(self):
        assert self.imgf is not None, "Error: Haven't seen a single depth image yet!"
        ptcloud = torch.zeros(1,3,self.ht,self.wd)
        if (self.imgf.shape[0] != int(self.ht) or self.imgf.shape[1] != int(self.wd)):
            depth = cv2.resize(self.imgf, (int(self.wd), int(self.ht)), interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
        else:
            depth = self.imgf
        ptcloud[0, 2].copy_(torch.FloatTensor(depth))  # Copy depth

        # Compute x & y values for the 3D points (= xygrid * depths)
        xy = ptcloud[:, 0:2]
        xy.copy_(self.intrinsics['xygrid'].expand_as(xy))  # = xygrid
        xy.mul_(ptcloud[0, 2])  # = xygrid * depths
        return ptcloud

#############################################################333
# Setup functions

#### Load checkpoint
def load_checkpoint(filename):
    # Load data from checkpoint
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        try:
            num_train_iter = checkpoint['num_train_iter']
        except:
            num_train_iter = checkpoint['epoch'] * checkpoint['args'].train_ipe
        print("=> loaded checkpoint (epoch: {}, num train iter: {})"
              .format(checkpoint['epoch'], num_train_iter))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        raise RuntimeError
    # Return
    return checkpoint, num_train_iter

#### Setup model with trained params
def setup_model(checkpoint, args, use_cuda=False):
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
    #                                       use_wt_sharpening=args.use_wt_sharpening,
    #                                       sharpen_start_iter=args.sharpen_start_iter,
    #                                       sharpen_rate=args.sharpen_rate, pre_conv=False,
    #                                       wide=args.wide_model, use_jt_angles=args.use_jt_angles,
    #                                       use_jt_angles_trans=args.use_jt_angles_trans,
    #                                       num_state=args.num_state_net)  # TODO: pre-conv
    #         posemaskpredfn = model.posemaskmodel.forward
    # else:
    model = ctrlnets.MultiStepSE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                           se3_type=args.se3_type, use_pivot=args.pred_pivot,
                                           use_kinchain=False,
                                           input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                           init_posese3_iden=args.init_posese3_iden,
                                           init_transse3_iden=args.init_transse3_iden,
                                           use_wt_sharpening=args.use_wt_sharpening,
                                           sharpen_start_iter=args.sharpen_start_iter,
                                           sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                                           decomp_model=args.decomp_model, wide=args.wide_model,
                                           use_jt_angles=args.use_jt_angles,
                                           use_jt_angles_trans=args.use_jt_angles_trans,
                                           num_state=args.num_state_net)
    posemaskpredfn = model.forward_pose_mask
    if use_cuda:
        model.cuda()  # Convert to CUDA if enabled

    # Update parameters from trained network
    try:
        model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluate mode
    model.eval()

    # Return model & fwd function
    return model, posemaskpredfn

def read_jt_angles(dataset, id, num_state=7):
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length

    # Setup memory
    sequence = data.generate_baxter_sequence(dataset, id)  # Get the file paths
    actconfigs = torch.FloatTensor(seq_len + 1, num_state)  # Actual data is same as state dimension

    # Load sequence
    t = torch.linspace(0, seq_len * step_len * (1.0 / 30.0), seq_len + 1).view(seq_len + 1, 1)  # time stamp
    for k in xrange(len(sequence)):
        # Get data table
        s = sequence[k]

        # Load configs
        state = data.read_baxter_state_file(s['state1'])
        actconfigs[k] = state['actjtpos']  # state dimension

    return {"actconfigs": actconfigs}

#### Dataset for getting test case
def setup_test_dataset(args):
    ########################
    ############ Get the data
    # Get datasets (TODO: Make this path variable)
    load_dir = '/home/barun/Projects/newdata/session_2017-9-12_201115/'

    # Get dimensions of ctrl & state
    try:
        statelabels, ctrllabels, trackerlabels = data.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
        print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
    except:
        statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
        ctrllabels = statelabels  # Just use the labels
        trackerlabels = []
        print("Could not read statectrllabels file. Reverting to labels in statelabels file")
    args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
    print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))

    # Find the IDs of the controlled joints in the state vector
    # We need this if we have state dimension > ctrl dimension and
    # if we need to choose the vals in the state vector for the control
    ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
    print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

    ## TODO: We have skipped checking for examples with left arm motion (or) right arm still
    valid_filter = lambda p, n, st, se: data.valid_data_filter(p, n, st, se,
                                                               mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                               reject_left_motion=False,
                                                               reject_right_still=False)
    baxter_data = data.read_recurrent_baxter_dataset(load_dir, 'sub',
                                                     step_len=args.step_len, seq_len=args.seq_len,
                                                     train_per=args.train_per, val_per=args.val_per,
                                                     valid_filter=valid_filter)
    disk_read_func = lambda d, i: read_jt_angles(d, i, num_state=args.num_state)
    test_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset

    return test_dataset, ctrlids_in_state

def create_config(angles):
    joints = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
    config = {joints[k]: angles[k] for k in xrange(len(angles))}
    return config

def get_angles(config):
    joints = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
    return torch.Tensor([float(config[joints[v]]) for v in xrange(len(joints))])

#############################################################
#### Pangolin viewer helpers

### Compute a point cloud give the arm config
# Assumes that a "Tensor" is the input, not a "Variable"
def generate_ptcloud(config, args):
    # Render the config & get the point cloud
    assert(not util.is_var(config))
    config_f = config.view(-1).clone().float()
    pts      = torch.FloatTensor(1, 3, args.img_ht, args.img_wd)
    labels   = torch.FloatTensor(1, 1, args.img_ht, args.img_wd)
    pangolin.render_arm(config_f.numpy(), pts[0].numpy(), labels[0].numpy())
    return pts.type_as(config), labels.type_as(config)

### Compute masks
def compute_masks_from_labels(labels, mesh_ids, args):
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

#############################################################333
############ SE3-Pose-Net control class
class SE3PoseSpaceOptimizer(object):
    def __init__(self, pargs):
        # Planning args
        self.pargs = pargs

        #######################
        # Create save directory and start tensorboard logger
        pargs.cuda = not pargs.no_cuda and torch.cuda.is_available()
        if pargs.save_dir == '':
            checkpoint_dir = pargs.checkpoint.rpartition('/')[0]
            pargs.save_dir = checkpoint_dir + '/planlogs/'
        print('Saving planning logs at: ' + pargs.save_dir)
        util.create_dir(pargs.save_dir)  # Create directory
        self.tblogger = util.TBLogger(pargs.save_dir + '/planlogs/')  # Start tensorboard logger

        # Set seed
        torch.manual_seed(pargs.seed)
        if pargs.cuda:
            torch.cuda.manual_seed(pargs.seed)

        # Default tensor type
        self.deftype = 'torch.cuda.FloatTensor' if pargs.cuda else 'torch.FloatTensor'  # Default tensor type

        ########################
        # Load pre-trained network
        self.checkpoint, self.num_train_iter = load_checkpoint(pargs.checkpoint)
        self.args = self.checkpoint['args']

        # BWDs compatibility
        if not hasattr(self.args, "use_gt_masks"):
            self.args.use_gt_masks, self.args.use_gt_poses = False, False
        if not hasattr(self.args, "num_state"):
            self.args.num_state = self.args.num_ctrl

        if not hasattr(self.args, "use_full_jt_angles"):
            self.args.use_full_jt_angles = True
        if self.args.use_full_jt_angles:
            self.args.num_state_net = self.args.num_state
        else:
            self.args.num_state_net = self.args.num_ctrl

        # Setup model
        self.model, self.posemaskfn = setup_model(self.checkpoint, self.args, self.pargs.cuda)

        # Sanity check some parameters (TODO: remove it later)
        assert (self.args.num_se3 == num_se3)
        if not pargs.real_robot:
            assert (self.args.img_scale == img_scale)

        ########################
        # Setup dataset
        self.dataset, self.ctrlids_in_state = setup_test_dataset(self.args)

    ## Get goal / start joint angles
    def setup_start_goal_angles(self):
        # Get start & goal samples
        start_id = self.pargs.start_id if (self.pargs.start_id >= 0) else np.random.randint(len(self.dataset))
        goal_id = start_id + round(self.pargs.goal_horizon / dt)
        print('Test dataset size: {}, Start ID: {}, Goal ID: {}, Duration: {}'.format(len(self.dataset),
                                                                                      start_id, goal_id,
                                                                                      self.pargs.goal_horizon))
        start_sample = self.dataset[start_id]
        goal_sample  = self.dataset[goal_id]

        # Get the joint angles
        start_full_angles = start_sample['actconfigs'][0]
        goal_full_angles = goal_sample['actconfigs'][0]
        start_angles = start_full_angles[self.ctrlids_in_state]
        goal_angles  = goal_full_angles[self.ctrlids_in_state]
        if self.pargs.only_top4_jts:
            assert not self.pargs.only_top6_jts, "Cannot enable control for 4 and 6 joints at the same time"
            print('Controlling only top 4 joints')
            goal_angles[4:] = start_angles[4:]
        elif self.pargs.only_top6_jts:
            assert not self.pargs.only_top4_jts, "Cannot enable control for 4 and 6 joints at the same time"
            print('Controlling only top 6 joints')
            goal_angles[6:] = start_angles[6:]
        elif self.pargs.ctrl_specific_jts is not '':
            print('Setting targets only for joints: {}. All other joints have zero error'
                  ' but can be controlled'.format(self.pargs.ctrl_specific_jts))
            ctrl_jts = [int(x) for x in self.pargs.ctrl_specific_jts.split(',')]
            for k in xrange(7):
                if k not in ctrl_jts:
                    goal_angles[k] = start_angles[k]

        # Initialize the pangolin viewer
        #start_pts, da_goal_pts = torch.zeros(1, 3, self.args.img_ht, self.args.img_wd), torch.zeros(1, 3, self.args.img_ht, self.args.img_wd)
        #pangolin.init_problem(start_angles.numpy(), goal_angles.numpy(), start_pts[0].numpy(), da_goal_pts[0].numpy())

        # Return
        return start_angles, goal_angles, start_id, goal_id, start_full_angles, goal_full_angles

    ## Compute pose & masks
    def predict_pose_masks(self, config, pts, tbtopic='Mask'):
        ## Predict pose / mask
        config_v = util.to_var(config.view(1,-1).type(self.deftype), requires_grad=False)
        if self.args.use_gt_masks:  # GT masks are provided!
            _, rlabels = generate_ptcloud(config, self.args)
            masks = util.to_var(compute_masks_from_labels(rlabels, self.args.mesh_ids, self.args))
            poses = self.posemaskfn([util.to_var(pts.type(self.deftype)), config_v])
        else:
            poses, masks = self.posemaskfn([util.to_var(pts.type(self.deftype)), config_v], train_iter=self.num_train_iter)

        # ## Display the masks as an image summary
        # maskdisp = torchvision.utils.make_grid(masks.data.cpu().view(-1, 1, self.args.img_ht, self.args.img_wd),
        #                                        nrow=self.args.num_se3, normalize=True, range=(0, 1))
        # info = {tbtopic: util.to_np(maskdisp.narrow(0, 0, 1))}
        # for tag, images in info.items():
        #     self.tblogger.image_summary(tag, images, 0)

        return poses, masks

    ### Function to generate the optimized control
    # Note: assumes that it get Variables
    def optimize_ctrl(self, poses, ctrl, goal_poses):

        # Do specific optimization based on the type
        if self.pargs.optimization == 'backprop':
            # Model has to be in training mode
            self.model.train()

            # ============ FWD pass + Compute loss ============#

            # FWD pass + loss
            poses_1 = util.to_var(poses.data, requires_grad=False)
            ctrl_1  = util.to_var(ctrl.data, requires_grad=True)
            _, pred_poses = self.model.transitionmodel([poses_1, ctrl_1])
            loss = self.args.loss_scale * ctrlnets.BiMSELoss(pred_poses, goal_poses)  # Get distance from goal

            # ============ BWD pass ============#

            # Backward pass & optimize
            self.model.zero_grad()  # Zero gradients
            zero_gradients(ctrl_1)  # Zero gradients for controls
            loss.backward()  # Compute gradients - BWD pass

            # Return
            return ctrl_1.grad.data.cpu().view(-1).clone(), loss.data[0]
        else:
            # No backprops here
            self.model.eval()

            # ============ Compute finite differenced controls ============#

            # Setup stuff for perturbation
            eps = self.pargs.gn_perturb
            nperturb = self.args.num_ctrl
            I = torch.eye(nperturb).type_as(ctrl.data)

            # Do perturbation
            poses_p = util.to_var(poses.data.repeat(nperturb + 1, 1, 1, 1))  # Replicate poses
            ctrl_p = util.to_var(ctrl.data.repeat(nperturb + 1, 1))  # Replicate controls
            ctrl_p.data[1:, :] += I * eps  # Perturb the controls

            # ============ FWD pass ============#

            # FWD pass
            _, pred_poses_p = self.model.transitionmodel([poses_p, ctrl_p])

            # Backprop only over the loss!
            pred_poses = util.to_var(pred_poses_p.data.narrow(0, 0, 1),
                                     requires_grad=True)  # Need grad of loss w.r.t true pred
            loss = self.args.loss_scale * ctrlnets.BiMSELoss(pred_poses, goal_poses)
            loss.backward()

            # ============ Compute Jacobian & GN-gradient ============#

            # Compute Jacobian
            Jt = pred_poses_p.data[1:].view(nperturb, -1).clone()  # nperturb x posedim
            Jt -= pred_poses_p.data.narrow(0, 0, 1).view(1, -1).expand_as(Jt)  # [ f(x+eps) - f(x) ]
            Jt.div_(eps)  # [ f(x+eps) - f(x) ] / eps

            ### Option 1: Compute GN-gradient using torch stuff by adding eps * I
            # This is incredibly slow at the first iteration
            Jinv = torch.inverse(torch.mm(Jt, Jt.t()) + self.pargs.gn_lambda * I)  # (J^t * J + \lambda I)^-1
            ctrl_grad = torch.mm(Jinv,
                                 torch.mm(Jt, pred_poses.grad.data.view(-1, 1)))  # (J^t*J + \lambda I)^-1 * (Jt * g)

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
            if self.pargs.gn_jac_check:
                # Set model in training mode
                self.model.train()

                # FWD pass
                poses_1 = util.to_var(poses.data, requires_grad=False)
                ctrl_1 = util.to_var(ctrl.data, requires_grad=True)
                _, pred_poses_1 = self.model.transitionmodel([poses_1, ctrl_1])
                pred_poses_1_v = pred_poses_1.view(1, -1)  # View it nicely

                ###
                # Compute Jacobian via multiple backward passes (SLOW!)
                Jt_1 = self.compute_jacobian(ctrl_1, pred_poses_1_v)
                diff_j = Jt.t() - Jt_1
                print('Jac diff => Min: {}, Max: {}, Mean: {}'.format(diff_j.min(), diff_j.max(), diff_j.abs().mean()))

                ###
                # Compute gradient via single backward pass + loss
                loss = self.args.loss_scale * ctrlnets.BiMSELoss(pred_poses_1, goal_poses)  # Get distance from goal
                self.model.zero_grad()  # Zero gradients
                zero_gradients(ctrl_1)  # Zero gradients for controls
                loss.backward()  # Compute gradients - BWD pass
                diff_g = ctrl_1.grad.data - torch.mm(Jt, pred_poses.grad.data.view(-1,
                                                                                   1))  # Error between backprop & J^T g from FD
                print('Grad diff => Min: {}, Max: {}, Mean: {}'.format(diff_g.min(), diff_g.max(), diff_g.abs().mean()))

            # Return the Gauss-Newton gradient
            return ctrl_grad.cpu().view(-1).clone(), loss.data[0]


def main():
    """Control using SE3-Pose-Nets
    """
    #############
    ## Parse arguments right at the top
    parser = argparse.ArgumentParser(description='Reactive control using SE3-Pose-Nets')

    # Required params
    parser.add_argument('-l', '--limb', dest='limb', required=True, choices=['left', 'right'],
                        help='limb on which to attach joint springs')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
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

    # TODO: Add criteria for convergence

    # Misc options
    parser.add_argument('--disp-freq', '-p', default=20, type=int,
                        metavar='N', help='print/disp/save frequency (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Display/Save options
    parser.add_argument('-s', '--save-dir', default='', type=str,
                        metavar='PATH', help='directory to save results in. (default: <checkpoint_dir>/planlogs/)')
    parser.add_argument('--real-robot', action='store_true', default=False,
                        help='use real robot! (default: False)')

    # Parse args
    pargs = parser.parse_args(rospy.myargv()[1:])

    ###### Setup ros node & jt space controller
    print("Initializing node... ")
    rospy.init_node("baxter_se3_control_%s" % (pargs.limb,), anonymous=True)

    # Create BAXTER controller
    jc = SE3ControlPositionMode(pargs.limb)
    rospy.on_shutdown(jc.clean_shutdown) # register shutdown callback

    # # If real robot, set left arm to default position
    # if pargs.real_robot:
    #     lc = SE3ControlPositionMode("left")
    #     print("Moving left arm to default position...")
    #     lc.move_to_pos({
    #         "left_e0": -0.309864,
    #         "left_e1":  1.55201,
    #         "left_s0":  0.341311,
    #         "left_s1":  0.0310631,
    #         "left_w0":  0.083602,
    #         "left_w1":  0.766607,
    #         "left_w2":  0.00076699,
    #     })

    ###### Setup controller (load trained net)
    se3optim = SE3PoseSpaceOptimizer(pargs)
    num_ctrl = se3optim.args.num_ctrl
    deftype  = se3optim.deftype

    # Initialize start & goal config
    start_angles, goal_angles, start_id, goal_id, \
    start_full_angles, goal_full_angles = se3optim.setup_start_goal_angles()
    curr_full_angles = start_full_angles.clone()

    #########
    # SOME TEST CONFIGS:
    # 66305 (works perfectly for all joints!)
    start_angles = torch.FloatTensor([ 0.0619, 0.1619, 1.1609, 0.9808, 0.3923, 0.6253, 0.0328])
    goal_angles  = torch.FloatTensor([4.1391e-01, -4.5127e-01, 8.9605e-01,  1.1968e+00, -4.4754e-05,  8.8374e-01, 6.2656e-02])

    # 31000 (Doesn't work :()
    #start_angles = torch.FloatTensor([0.5052, -0.4135, 1.0945,  1.6024,  1.0821, -0.6957, -0.2535])
    #goal_angles  = torch.FloatTensor([0.0880, -0.3266, 0.8092,  1.1611,  0.2845,  0.5481, -0.4666])

    # 41000 (Kinda OK, joint 4 still stays at errors of ~7 degrees)
    #start_angles = torch.FloatTensor([-0.6730,  0.5814,  1.3403,  1.7309, -0.4106,  0.4301, -1.7868])
    #goal_angles  = torch.FloatTensor([-0.0374, -0.2891,  1.2771,  1.4422, -0.4017,  0.9142, -0.7823])

    # 51000 (Has issues with 4,5)
    #start_angles = torch.FloatTensor([ 0.3525, -0.1269,  1.1656,  1.3804,  0.1220,  0.3742, -0.1250])
    #goal_angles  = torch.FloatTensor([ 0.4959, -0.2184,  1.2100,  1.8197,  0.3975, -0.7801,  0.2076])

    # Print error
    print('Initial jt angle error:')
    full_deg_error = (start_angles - goal_angles) * (180.0 / np.pi)  # Full error in degrees
    print(full_deg_error.view(se3optim.args.num_ctrl, 1))

    ###### Setup subscriber to depth images if using real robot
    if pargs.real_robot:
        print("Setting up depth subscriber...")
        depth_subscriber = DepthImageSubscriber(se3optim.args.img_ht,
                                                se3optim.args.img_wd,
                                                se3optim.args.img_scale,
                                                se3optim.args.cam_intrinsics)
        time.sleep(1)

    ###### Goal stuff
    # Set goal position, get observation
    jc.move_to_pos(create_config(goal_angles.numpy()))
    if pargs.real_robot:
        goal_pts = depth_subscriber.get_ptcloud()
    else:
        goal_pts, _ = generate_ptcloud(goal_angles, se3optim.args)

    # Compute goal poses
    if se3optim.args.use_full_jt_angles:
        curr_full_angles[se3optim.ctrlids_in_state] = goal_angles # Only the controlled joints move
        goal_poses, goal_masks = se3optim.predict_pose_masks(curr_full_angles, goal_pts, 'Goal Mask')
    else:
        goal_poses, goal_masks = se3optim.predict_pose_masks(goal_angles, goal_pts, 'Goal Mask')

    ###### Start stuff
    # Set start position, get observation
    jc.move_to_pos(create_config(start_angles.numpy()))
    if pargs.real_robot:
        start_pts = depth_subscriber.get_ptcloud()
    else:
        start_pts, _ = generate_ptcloud(start_angles, se3optim.args)

    # Compute goal poses
    if se3optim.args.use_full_jt_angles:
        curr_full_angles[se3optim.ctrlids_in_state] = start_angles # Only the controlled joints move
        start_poses, start_masks = se3optim.predict_pose_masks(curr_full_angles, start_pts, 'Start Mask')
    else:
        start_poses, start_masks = se3optim.predict_pose_masks(start_angles, start_pts, 'Start Mask')

    # Render the poses
    # NOTE: Data passed into cpp library needs to be assigned to specific vars, not created on the fly (else gc will free it)
    start_masks_f, goal_masks_f = start_masks.data.cpu().float(), goal_masks.data.cpu().float()
    _, start_labels = start_masks_f.max(dim=1); start_labels_f = start_labels.float()
    _, goal_labels  = goal_masks_f.max(dim=1);  goal_labels_f  = goal_labels.float()
    start_poses_f, goal_poses_f = start_poses.data.cpu().float(), goal_poses.data.cpu().float()
    pangolin.update_real_init(start_angles.numpy(),
                              start_pts[0].numpy(),
                              start_poses_f[0].numpy(),
                              start_labels_f.numpy(),
                              goal_angles.numpy(),
                              goal_pts[0].numpy(),
                              goal_poses_f[0].numpy(),
                              goal_labels_f.numpy())

    ###### Run the controller
    # Init stuff
    ctrl_mag = pargs.max_ctrl_mag
    angles, deg_errors = torch.zeros(pargs.max_iter + 1, 7), torch.zeros(pargs.max_iter + 1, 7)
    angles[0], deg_errors[0] = start_angles, full_deg_error
    ctrl_grads, ctrls = torch.zeros(pargs.max_iter, num_ctrl), \
                        torch.zeros(pargs.max_iter, num_ctrl)
    losses = torch.zeros(pargs.max_iter)

    # Init vars for all items
    init_ctrl_v  = util.to_var(torch.zeros(1, num_ctrl).type(se3optim.deftype), requires_grad=True)  # Need grad w.r.t this
    goal_poses_v = util.to_var(goal_poses.data, requires_grad=False)

    # Plots for errors and loss
    fig, axes = plt.subplots(2, 1)
    fig.show()

    # Run the controller
    gen_time, posemask_time, optim_time, viz_time, rest_time = AverageMeter(), AverageMeter(), AverageMeter(), \
                                                               AverageMeter(), AverageMeter()
    for it in xrange(pargs.max_iter):
        # Print
        print('\n #####################')

        # Get current observation
        start = time.time()

        curr_angles = get_angles(jc.cur_pos())
        if pargs.real_robot:
            curr_pts = depth_subscriber.get_ptcloud()
        else:
            curr_pts, _ = generate_ptcloud(curr_angles, se3optim.args)
        curr_pts = curr_pts.type(deftype)

        gen_time.update(time.time() - start)

        # Predict poses and masks
        start = time.time()
        if se3optim.args.use_full_jt_angles:
            curr_full_angles[se3optim.ctrlids_in_state] = goal_angles  # Only the controlled joints move
            curr_poses, curr_masks = se3optim.predict_pose_masks(curr_full_angles, curr_pts, 'Curr Mask')
        else:
            curr_poses, curr_masks = se3optim.predict_pose_masks(curr_angles, curr_pts, 'Curr Mask')
        curr_poses_f, curr_masks_f = curr_poses.data.cpu().float(), curr_masks.data.cpu().float()

        posemask_time.update(time.time() - start)

        # Render poses and masks using Pangolin
        start = time.time()

        _, curr_labels = curr_masks_f.max(dim=1)
        curr_pts_f = curr_pts.cpu().float()
        curr_labels_f = curr_labels.float()
        pangolin.update_real_curr(curr_angles.numpy(),
                                  curr_pts_f[0].numpy(),
                                  curr_poses_f[0].numpy(),
                                  curr_labels_f.numpy())

        viz_time.update(time.time() - start)

        # Run one step of the optimization (controls are always zeros, poses change)
        start = time.time()

        ctrl_grad, loss = se3optim.optimize_ctrl(curr_poses,
                                                 init_ctrl_v,
                                                 goal_poses=goal_poses_v)
        ctrl_grads[it] = ctrl_grad.cpu().float()  # Save this

        # Set last 3 joint's controls to zero
        if pargs.only_top4_jts:
            ctrl_grad[4:] = 0
        elif pargs.only_top6_jts:
            ctrl_grad[6:] = 0

        optim_time.update(time.time() - start)

        # Get the control direction and scale it by max control magnitude
        start = time.time()

        if ctrl_mag > 0:
            ctrl_dirn = ctrl_grad.cpu().float() / ctrl_grad.norm(2)  # Dirn
            curr_ctrl = ctrl_dirn * ctrl_mag  # Scale dirn by mag
            ctrl_mag *= pargs.ctrl_mag_decay  # Decay control magnitude
        else:
            curr_ctrl = ctrl_grad.cpu().float()

        # Apply control (simple velocity integration)
        next_angles = curr_angles - (curr_ctrl * dt)
        jc.move_to_pos(create_config(next_angles))

        # Save stuff
        losses[it] = loss
        ctrls[it] = curr_ctrl
        angles[it + 1] = next_angles
        deg_errors[it + 1] = (next_angles - goal_angles) * (180.0 / np.pi)

        # Print losses and errors
        print('Control Iter: {}/{}, Loss: {}'.format(it + 1, pargs.max_iter, loss))
        print('Joint angle errors in degrees: ',
              torch.cat([deg_errors[it + 1].unsqueeze(1), full_deg_error.unsqueeze(1)], 1))

        # Plot the errors & loss
        if (it % 4) == 0:
            axes[0].set_title("Iter: {}, Jt angle errors".format(it + 1))
            axes[0].plot(deg_errors.numpy()[:it + 1])
            axes[1].set_title("Iter: {}, Loss".format(it + 1))
            axes[1].plot(losses.numpy()[:it + 1])
            fig.canvas.draw()  # Render
            plt.pause(0.01)
        if (it % se3optim.args.disp_freq) == 0:  # Clear now and then
            for ax in axes:
                ax.cla()

        # Finish
        rest_time.update(time.time() - start)
        print('Gen: {:.3f}({:.3f}), PoseMask: {:.3f}({:.3f}), Viz: {:.3f}({:.3f}),'
              ' Optim: {:.3f}({:.3f}), Rest: {:.3f}({:.3f})'.format(
            gen_time.val, gen_time.avg, posemask_time.val, posemask_time.avg,
            viz_time.val, viz_time.avg, optim_time.val, optim_time.avg,
            rest_time.val, rest_time.avg))

    # Print final stats
    print('=========== FINISHED ============')
    print('Final loss after {} iterations: {}'.format(pargs.max_iter, losses[-1]))
    print('Final angle errors in degrees: ')
    print(deg_errors[-1].view(7, 1))

    # Save stats and exit
    stats = {'args': se3optim.args, 'pargs': pargs, 'start_id': start_id,
             'goal_id': goal_id, 'start_angles': start_angles, 'goal_angles': goal_angles,
             'angles': angles, 'ctrls': ctrls, 'predctrls': ctrl_grads, 'deg_errors': deg_errors,
             'losses': losses}
    torch.save(stats, pargs.save_dir + '/planstats.pth.tar')


if __name__ == "__main__":
    main()
