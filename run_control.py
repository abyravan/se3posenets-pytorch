# Global imports
import argparse
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

##########
# Load the pangolin visualizer library using cffi
from cffi import FFI
ffi = FFI()
pangolin = ffi.dlopen('/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/torchviz/build/libtorchviz_gd_baxter.so')
ffi.cdef('''
    void initialize_viz(int seqLength, int imgHeight, int imgWidth, float imgScale, int nSE3,
                        float fx, float fy, float cx, float cy, float dt, int oldgrippermodel,
                        const char *savedir);
    void terminate_viz();
    float** initialize_problem(const float *input_jts, const float *target_jts);
    void update_viz(const float *inputpts, const float *outputpts_gt, 
                    const float *outputpts_pred, const float *jtangles_gt, 
                    const float *jtangles_pred);
    float ** render_arm(const float *config);
    void update_da(const float *input_da, const float *target_da, const float *target_da_ids, 
                   const float *warpedtarget_da, const float *target_da);
    float ** compute_gt_da(const float *input_jts, const float *target_jts, const int winsize, 
                           const float thresh, const float *final_pts);
    float ** update_pred_pts(const float *net_preds, const float *net_grads);
    float ** update_pred_pts_unwarped(const float *net_preds, const float *net_grads);
    void initialize_poses(const float *init_poses, const float *tar_poses);
    void update_masklabels_and_poses(const float *curr_masks, const float *curr_poses);
    void start_saving_frames(const char *framesavedir);
    void stop_saving_frames();
''')

# TODO: Make this cleaner, we don't need most of these parameters to create the pangolin window
img_ht, img_wd, img_scale = 240, 320, 1e-4
seq_len = 1 # For now, only single step
num_se3 = 8 # TODO: Especially this parameter!
dt = 1.0/30.0
oldgrippermodel = False # TODO: When are we actually going to use the new ones?
cam_intrinsics = {'fx': 589.3664541825391 / 2,
                  'fy': 589.3664541825391 / 2,
                  'cx': 320.5 / 2,
                  'cy': 240.5 / 2}
savedir = 'temp' # TODO: Fix this!

# Create the pangolin window
pangolin.initialize_viz(seq_len, img_ht, img_wd, img_scale, num_se3,
                        cam_intrinsics['fx'], cam_intrinsics['fy'],
                        cam_intrinsics['cx'], cam_intrinsics['cy'],
                        dt, oldgrippermodel, savedir)

##########
# NOTE: When importing torch before initializing the pangolin window, I get the error:
#   Pangolin X11: Unable to retrieve framebuffer options
# Long story short, initializing torch before pangolin messes things up big time.
# Also, only the conda version of torch works otherwise there's an issue with loading the torchviz library before torch
#   ImportError: dlopen: cannot load any more object with static TLS

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import util
from util import AverageMeter

##########
# Parse arguments
parser = argparse.ArgumentParser(description='Reactive control using SE3-Pose-Nets')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for training (default: none)')

# Problem options
parser.add_argument('--start-id', default=-1, type=int, metavar='N',
                    help='ID in the test dataset for the start state (default: -1 = randomly chosen)')
parser.add_argument('--target-horizon', default=1.5, type=float, metavar='SEC',
                    help='Planning target horizon in seconds (default: 1.5)')

# Planner options
parser.add_argument('--optimization', default='gn', type=str, metavar='OPTIM',
                    help='Type of optimizer to use: [gn] | backprop')
parser.add_argument('--max-iter', default=100, type=int, metavar='N',
                    help='Maximum number of planning iterations (default: 100)')
parser.add_argument('--gn-perturb', default=1e-3, type=float, metavar='EPS',
                    help='Perturbation for the finite-differencing to compute the jacobian (default: 1e-3)')
parser.add_argument('--gn-lambda', default=1e-4, type=float, metavar='LAMBDA',
                    help='Damping constant (default: 1e-4)')
parser.add_argument('--gn-grad-check', action='store_true', default=False,
                    help='check GN gradient against the backprop gradient (default: False)')
parser.add_argument('--gn-jac-check', action='store_true', default=False,
                    help='check FD jacobian against the analytical jacobian (default: False)')
parser.add_argument('--max-ctrl-mag', default=1.0, type=float, metavar='UMAX',
                    help='Maximum allowable control magnitude (default: 1 rad/s)')
parser.add_argument('--ctrl-mag-decay', default=0.99, type=float, metavar='W',
                    help='Decay the control magnitude by scaling by this weight after each iter (default: 0.99)')

# TODO: Add criteria for convergence

# Misc options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Display/Save options
parser.add_argument('--disp-freq', '-p', default=20, type=int,
                    metavar='N', help='print/disp/save frequency (default: 10)')
parser.add_argument('-s', '--save-dir', default='results', type=str,
                    metavar='PATH', help='directory to save results in. If it doesnt exist, will be created. (default: results/)')

def main():
    # Parse args
    global pargs, num_train_iter
    pargs = parser.parse_args()
    pargs.cuda = not pargs.no_cuda and torch.cuda.is_available()

    # Create save directory and start tensorboard logger
    util.create_dir(pargs.save_dir)  # Create directory
    tblogger = util.TBLogger(pargs.save_dir + '/planlogs/')  # Start tensorboard logger

    # Set seed
    torch.manual_seed(pargs.seed)
    if pargs.cuda:
        torch.cuda.manual_seed(pargs.seed)

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

    # Create a model
    model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                  se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                  input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                  init_posese3_iden=False, init_transse3_iden=False,
                                  use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                                  sharpen_rate=args.sharpen_rate, pre_conv=False) # TODO: pre-conv
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    # Update parameters from trained network
    try:
        model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Sanity check some parameters (TODO: remove it later)
    assert(args.num_se3 == num_se3)
    assert(args.seq_len == seq_len)
    assert(args.img_scale == img_scale)
    try:
        cam_i = args.cam_intrinsics
        for _, key in enumerate(cam_intrinsics):
            assert(cam_intrinsics[key] == cam_i[key])
    except AttributeError:
        args.cam_intrinsics = cam_intrinsics # In case it doesn't exist

    ########################
    ############ Get the start/goal

    time.sleep(100)

################ RUN MAIN
if __name__ == '__main__':
    main()