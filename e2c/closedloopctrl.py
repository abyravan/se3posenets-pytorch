#!/usr/bin/env python
# To run:
# sourceblocks && python e2c/closedloopctrl.py -c <yaml-file>

# Global imports
import h5py
import numpy as np
import os
import sys
import argparse, configargparse

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import _init_paths
import util
import blockdata as BD
import e2c.model as e2cmodel
import e2c.helpers as e2chelpers

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gazebo_learning_planning.srv import Configure
from gazebo_learning_planning.srv import ConfigureRequest
from gazebo_learning_planning.msg import ConfigureObject

# Chris's simulator
from simulator.yumi_simulation import YumiSimulation

# Setup arm link IDs
arm_l_idx = [0, 2, 4, 6, 8, 10, 12]
arm_r_idx = [1, 3, 5, 7, 9, 11, 13]
gripper_r_idx = 14
gripper_l_idx = 15

################ Interface to YUMI robot
class YUMIInterface(object):
    #### Callbacks defined before constructor initialization
    # Joint state callback
    def _js_cb(self, msg):
        self.q     = np.array(msg.position)
        self.dq    = np.array(msg.velocity)
        for name, q in zip(msg.name, msg.position):
            self.qdict[name] = q

    # RGB image callback
    def _rgb_cb(self, msg):
        try:
            frame    = self.bridge.imgmsg_to_cv2(msg)
            self.rgb = np.array(frame, dtype=np.uint8)
        except CvBridgeError as e:
            print(e)

    # Depth image callback
    def _depth_cb(self, msg):
        try:
            frame      = self.bridge.imgmsg_to_cv2(msg)
            self.depth = np.array(frame, dtype=np.float32)
        except CvBridgeError as e:
            print(e)

    #### Constructor
    def __init__(self, hz=30.):
        # Setup service proxy for initializing simulator state from h5
        self.configure = rospy.ServiceProxy("simulation/configure", Configure)

        # Setup callbacks and services for JointState
        self.js_sub = rospy.Subscriber("/robot/joint_states", JointState, self._js_cb)
        self.js_cmd = rospy.Publisher(YumiSimulation.listen_topic, JointState, queue_size=1000)
        self.q      = None
        self.dq     = None
        self.qdict  = {}
        self.hz     = hz

        # Setup callback for RGB/D images
        self.bridge    = CvBridge()
        self.rgb_sub   = rospy.Subscriber("/robot/image", Image, self._rgb_cb)
        self.depth_sub = rospy.Subscriber("/robot/depth_image", Image, self._depth_cb)
        self.rgb       = None
        self.depth     = None

    # Send a full command to the
    def commandJts(self, rg, lg, ra, la):
        '''
        Publish the provided position commands.
        '''
        data = {}
        for k, v in zip(YumiSimulation.robot_left_gripper, [lg]):
            data[k] = v
        for k, v in zip(YumiSimulation.robot_right_gripper, [rg]):
            data[k] = v
        for k, v in zip(YumiSimulation.yumi_left_arm, la):
            data[k] = v
        for k, v in zip(YumiSimulation.yumi_right_arm, ra):
            data[k] = v
        msg          = JointState()
        msg.name     = data.keys()
        msg.position = data.values()
        self.js_cmd.publish(msg)

    # Command joint velocities to the robot (internally integrates to get positions and sends those)
    def commandJtVelocities(self, cmd, dt=None):
        if dt is None:
            dt = 1./self.hz
        msg          = JointState()
        msg.name     = YumiSimulation.yumi_joint_names
        msg.position = self.q + cmd*dt
        self.js_cmd.publish(msg)

    # Replay the commands from a h5 data file
    def replayH5(self, h5, start=None, goal=None):
        # Get the commands from the h5 data
        rg = np.array(h5['right_gripper_cmd'])
        lg = np.array(h5['left_gripper_cmd'])
        ra = np.array(h5['right_arm_cmd'])
        la = np.array(h5['left_arm_cmd'])

        # Get start and goal ids
        start = 0 if start is None else max(start, 0) # +ve start id
        if goal is not None:
            assert (goal >= start), "Goal id: {} is < Start id: {}".format(goal, start)
        else:
            goal = rg.shape[0]
        print('Replaying the commands from the H5 file. Start: {}, Goal: {}'.format(start, goal))

        # Initialize with the first command
        rospy.sleep(0.5)
        self.commandJts(rg[start], lg[start], ra[start], la[start])
        rospy.sleep(1.)

        # Send all commands from [start, goal)
        rate = rospy.Rate(self.hz)
        for i in range(start, goal):
            # publish a single command to the robot
            self.commandJts(rg[i], lg[i], ra[i], la[i])
            rate.sleep()

    # Initialize the state of the simulator from a H5 data file
    def configureH5(self, h5):
        # Setup object configuration. Max of 20 objects we might actually care about
        msg = ConfigureRequest()
        print('Setting up the simulator state from the initial config of the H5 file')
        for i in range(20):
            name = "pose%d" % i
            if name in h5:
                poses = np.array(h5[name])
                if poses.shape[0] > 0:
                    # Setup object pose
                    pose = poses[0]
                    obj = ConfigureObject()
                    obj.id.data = i
                    obj.pose.position.x = pose[0]
                    obj.pose.position.y = pose[1]
                    obj.pose.position.z = pose[2]
                    obj.pose.orientation.x = pose[3]
                    obj.pose.orientation.y = pose[4]
                    obj.pose.orientation.z = pose[5]
                    obj.pose.orientation.w = pose[6]
                    # Add object pose to msg
                    msg.object_poses.append(obj)

        # Setup robot configuration
        q = np.array(h5["robot_positions"])[0]
        msg.joint_state.position = q

        # Send configuration to simulator
        self.configure(msg)

################ Control arguments
def setup_control_options():

    ## Parse arguments right at the top
    parser = configargparse.ArgumentParser(description='Closed-loop control using Enc-Trans-Dec networks')

    # Required params
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                        help='path to saved network to use for training (default: none)')

    # Problem options
    parser.add_argument('--num-configs', type=int, default=10, metavar='N',
                        help='Num configs to test (default: 10)')
    parser.add_argument('--data-key', default='val', type=str,
                        help='Run tests on this dataset: train | [val] | test')

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

    # Misc options
    parser.add_argument('--disp-freq', '-p', default=20, type=int,
                        metavar='N', help='print/disp/save frequency (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA testing (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Display/Save options
    parser.add_argument('-s', '--save-dir', default='', type=str,
                        metavar='PATH', help='directory to save results in. (default: <checkpoint_dir>/planlogs/)')

    # Return
    return parser

def load_checkpoint(path, use_cuda=True):
    if os.path.isfile(path):
        ### Load checkpoint
        print("=> [MODEL] Loading model from checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        args = checkpoint['args']
        assert (use_cuda == args.cuda), "Mismatch in CUDA options in planning arguments & saved checkpoint arguments."

        ### Display stuff
        best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
        best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
        print("=> [MODEL] Epoch {}, Train iter {}, Best validation loss: {} was from epoch: {}"
              .format(checkpoint['epoch'], checkpoint['train_iter'],
                      best_loss, best_epoch))

        ### Load the model
        if args.deterministic:
            print('[MODEL] Using deterministic model')
            assert (args.varkl_wt == 0), "Deterministic model cannot have varkl-wt > 0"
            modelfn = e2cmodel.DeterministicModel
        else:
            print('[MODEL] Using probabilistic model')
            modelfn = e2cmodel.E2CModel
        model = modelfn(
            enc_img_type=args.enc_img_type, dec_img_type=args.dec_img_type,
            enc_inp_state=args.enc_inp_state, dec_pred_state=args.dec_pred_state,
            conv_enc_dec=args.conv_enc_dec, dec_pred_norm_rgb=args.dec_pred_norm_rgb,
            trans_setting=args.trans_setting, trans_pred_deltas=args.trans_pred_deltas,
            trans_model_type=args.trans_model_type, state_dim=args.num_ctrl,
            ctrl_dim=args.num_ctrl, wide_model=args.wide_model, nonlin_type=args.nonlin_type,
            norm_type=args.norm_type, coord_conv=args.coord_conv, img_size=(args.img_ht, args.img_wd))
        if args.cuda:
            model.cuda()  # Convert to CUDA if enabled

        # Print number of trainable parameters in model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[MODEL] Number of parameters: {}'.format(num_params))

        # Update model params with trained checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # Load network

        # Return model & args
        return model, args
    else:
        assert False, "[MODEL] No checkpoint found at '{}'"

# Get IDs of the states at which the "label" changes
def get_label_changes(h5):
    # Get the IDs of the states where the labels change
    label   = np.array(h5["label"])
    changed = np.abs(np.diff(label, axis=-1)) > 0
    selected     = np.zeros_like(label)
    selected[1:] = changed
    selected[-1] = 1
    selected[0]  = 1
    ids = np.nonzero(selected)[0]

    # Get the labels
    labelstrings = np.array(h5["label_to_string"])
    idlabels = labelstrings[label[ids]]

    # Return ids of label changes and corresponding labels
    return ids, idlabels

################ Controller using trained network
def run_local_controller(goaldata, model, args, pargs):
    # todo: Get latest RGB/Depth image, combine, get latest state, run fwd pass,
    # todo: compute error, get gradients, update control, execute, plot stuff,
    # todo: check convergence, accumulate stats
    stats = {}
    return stats

################ MAIN
#@profile
def main():
    # Parse args
    parser = setup_control_options()
    pargs  = parser.parse_args(rospy.myargv()[1:])
    pargs.cuda = (not pargs.no_cuda) and torch.cuda.is_available()

    ###### Setup ros node & jt space controller
    print("Initializing ROS node... ")
    rospy.init_node("yumi_closed_loop_ctrl", anonymous=False)

    # Create save directory and start tensorboard logger
    util.create_dir(pargs.save_dir)  # Create directory
    tblogger = util.TBLogger(pargs.save_dir)  # Start tensorboard logger

    # Create logfile to save prints
    logfile = open(pargs.save_dir + '/logfile.txt', 'w')
    backup  = sys.stdout
    sys.stdout = util.Tee(sys.stdout, logfile)

    ########################
    ############ Load model, controller & datasets
    # Set seed
    torch.manual_seed(pargs.seed)
    np.random.seed(pargs.seed)
    if pargs.cuda:
        torch.cuda.manual_seed(pargs.seed)

    # Load model
    model, args = load_checkpoint(pargs.checkpoint, pargs.cuda)
    assert (args.enc_img_type == 'rgbd'), "Currently the control code is implemented only for " \
                                          "networks with RGBD data input"

    # Load YUMI interface
    interface = YUMIInterface(hz=30.)

    #### Setup h5 datasets for control
    args.remove_static_examples = False # Shouldn't change the train/val/test h5 files
    train_dataset, _, _ = e2chelpers.parse_options_and_setup_block_dataset_loader(args)

    # Get the H5 files for the corresponding dataset (train/val/test)
    assert (pargs.data_key in ['train', 'val', 'test']),\
        "Unknown data key input: {}".format(pargs.data_key)
    datafiles = []
    for d in train_dataset.datasets:
        st, ed = d['files'][pargs.data_key]
        datafiles += [os.path.join(d['path'], h5file) for h5file in d['files']['names'][st:ed]]

    #### Run the control tests
    stats = {'ids': [], 'files': [], 'configstats': [], 'configoptsteps': [],
             'configoptsuccsteps': []}
    for k in range(pargs.num_configs):
        # Sample an example h5
        id = np.random.randint(0, len(datafiles))
        if id in stats['ids']:
            continue
        stats['ids'].append(id) # Add ID to samples
        stats['files'].append(datafiles[id])

        # Get data from h5 file
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with h5py.File(datafiles[id], 'r') as h5data:
            # Setup the config on the simulator
            interface.configureH5(h5data)

            # TODO: Sample a start/goal from the h5file
            # TODO: Either set a sub-goal every "k" steps till end of file (or)
            # TODO: Set subgoal based on transitions (or)
            # TODO: Set a single subgoal for each h5 within transition with a fixed horizon (or)
            # TODO: Set a single subgoal across transitions
            subgoalids, subgoallabels = get_label_changes(h5data)
            numsteps = len(subgoalids)-1
            stepstats = {'status': torch.zeros(numsteps), 'optimstats': []}
            for j in range(numsteps):
                # Get the start/goal pair
                stid, glid = subgoalids[j], subgoalids[j+1]
                stlb, gllb = str(subgoallabels[j]), str(subgoallabels[j+1])

                # If it is a gripper action, just play back the msgs
                if ('close_gripper' in stlb) or ('release' in stlb):
                    print('Test: {}/{}, Step: {}/{}, Reached gripper action, playing back the messages'.format(
                        k+1, pargs.num_configs, j+1, numsteps))
                    interface.replayH5(h5data, stid, glid)
                    stepstats['status'][j] = 2 # Gripper action
                    continue

                # Else run the controller to get from start to goal!
                print('Test: {}/{}, Step: {}/{}, Testing the controller'.format(
                    k+1, pargs.num_configs, j+1, numsteps))

                # Get the goal images & state data (targets for the optimization)
                goalrgb   = BD.NumpyBHWCToTorchBCHW(BD.ConvertPNGListToNumpy(
                    h5data['images_rgb'][glid:glid+1])).float() / 255.0
                goaldepth = BD.NumpyBHWToTorchBCHW(BD.ConvertDepthPNGListToNumpy(
                    h5data['images_depth'][glid:glid+1])).float() / 3.0
                goalimg   = torch.cat([goalrgb, goaldepth], 1).to(device) # 1 x 4 x ht x wd
                goalstate = torch.from_numpy(h5data['robot_positions'][glid:glid+1][:, arm_r_idx]).float().to(device)

                # Run the optimization with the input parameters
                optimstats = run_local_controller((goalimg, goalstate), model, args, pargs)
                print('Final error after optimization (Opt/Jt/Img): {}/{}/{}'.format(
                    optimstats['finalopterr'], optimstats['finaljterr'], optimstats['finalimgerr']))
                if (optimstats['success']):
                    print('Test: {}/{}, Step: {}/{}, Local controller converged in {} iterations. '
                          .format(k+1, pargs.num_configs, j+1, numsteps, optimstats['iters']))
                    stepstats['status'][j] = 1 # Success
                else:
                    print('Test: {}/{}, Step: {}/{}, Local controller failed to converge after {} iterations. '
                          .format(k+1, pargs.num_configs, j+1, numsteps, optimstats['iters']))
                    stepstats['status'][j] = 0 # Failure

                    # Execute the command to reach the goal of that step (to ensure continuity)
                    # TODO: Maybe reset to initial pos of that state and redo the traj??
                    interface.replayH5(h5data, glid, glid) # Reach the goal position
                    rospy.sleep(1.)

                # Save optimstats and continue
                stepstats['optimstats'].append(optimstats)

            # Print stats
            noptimsteps, noptimsuccsteps = int(stepstats['status'].ne(2).sum()), int(stepstats['status'].eq(1).sum())
            print('Test: {}/{}, Steps converged: {}/{} ({}%)'.format(
                k+1, pargs.num_configs, noptimsuccsteps, noptimsteps, noptimsuccsteps * (100.0/noptimsteps)))
            stats['configstats'].append(stepstats)
            stats['configoptsteps'].append(noptimsteps)
            stats['configoptsuccsteps'].append(noptimsuccsteps)

    ### Print final stats and save stuff
    # TODO: Print other nice stuff like errors etc.
    succper = torch.Tensor([s*(100.0/n) for s,n in zip(stats['configoptsuccsteps'], stats['configoptsteps'])])
    print('Mean/Std/Median success percentage: {}/{}/{}'.format(succper.mean().item(), succper.std().item(),
                                                                succper.median().item()))

if __name__ == "__main__":
    # TODO(@cpaxton):
    if len(sys.argv) < 2:
        print("usage: %s [filename]" % str(sys.argv[0]))

    rospy.init_node("control_robot")
    filename = sys.argv[1]
    h5 = h5py.File(os.path.expanduser(filename))
    iface = YUMIInterface(hz=30.)
    iface.configureH5(h5)
    iface.replayH5(h5)
