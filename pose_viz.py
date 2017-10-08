#!/usr/bin/env python

# Global imports
import _init_paths

# Load pangolin visualizer library
from torchviz import pangoposeviz
from torchviz import realctrlviz # Had to add this to get the file to work, otherwise gave static TLS error
pangolin = pangoposeviz.PyPangolinPoseViz()

##########
# NOTE: When importing torch before initializing the pangolin window, I get the error:
#   Pangolin X11: Unable to retrieve framebuffer options
# Long story short, initializing torch before pangolin messes things up big time.
# Also, only the conda version of torch works otherwise there's an issue with loading the torchviz library before torch
#   ImportError: dlopen: cannot load any more object with static TLS
# With the new CUDA & NVIDIA drivers the new conda also doesn't work. had to move to CYTHON to get code to work

# Torch imports
import torch
import torch.optim
import torch.utils.data

# Local imports
import data
import ctrlnets
import util

# Other imports
import argparse
import os

##########
# Parse arguments
parser = argparse.ArgumentParser(description='Reactive control using SE3-Pose-Nets')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for training (default: none)')

def main():
    # Parse args
    global pargs, args, num_train_iter
    pargs = parser.parse_args()
    pargs.cuda = True

    # Default tensor type
    deftype = 'torch.cuda.FloatTensor' if pargs.cuda else 'torch.FloatTensor' # Default tensor type

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
    if not hasattr(args, "use_gt_masks"):
        args.use_gt_masks, args.use_gt_poses = False, False
    if not hasattr(args, "num_state"):
        args.num_state = args.num_ctrl
    if not hasattr(args, "use_gt_angles"):
        args.use_gt_angles, args.use_gt_angles_trans = False, False
    if not hasattr(args, "num_state"):
        args.num_state = 7

    ## TODO: Either read the args right at the top before calling pangolin - might be easier, somewhat tricky to do BWDs compatibility
    ## TODO: Or allow pangolin to change the args later

    # Create a model
    if args.seq_len == 1:
        if args.use_gt_masks:
            model = ctrlnets.SE3OnlyPoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                              se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                              input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                              init_posese3_iden=False, init_transse3_iden=False,
                                              use_wt_sharpening=args.use_wt_sharpening,
                                              sharpen_start_iter=args.sharpen_start_iter,
                                              sharpen_rate=args.sharpen_rate, pre_conv=False,
                                              wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                                              use_jt_angles_trans=args.use_jt_angles_trans,
                                              num_state=args.num_state)  # TODO: pre-conv
            posemaskpredfn = model.posemodel.forward
        elif args.use_gt_poses:
            assert False, "No need to run tests with GT poses provided"
        else:
            model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                          se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                          input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                          init_posese3_iden=False, init_transse3_iden=False,
                                          use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                                          sharpen_rate=args.sharpen_rate, pre_conv=False, wide=args.wide_model) # TODO: pre-conv
            posemaskpredfn = model.posemaskmodel.forward
    else:
        if args.use_gt_masks:
            modelfn = ctrlnets.MultiStepSE3OnlyPoseModel
        elif args.use_gt_poses:
            assert False, "No need to run tests with GT poses provided"
        else:
            modelfn = ctrlnets.MultiStepSE3PoseModel
        model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                        se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                        input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                        init_posese3_iden=args.init_posese3_iden,
                        init_transse3_iden=args.init_transse3_iden,
                        use_wt_sharpening=args.use_wt_sharpening,
                        sharpen_start_iter=args.sharpen_start_iter,
                        sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                        decomp_model=args.decomp_model, wide=args.wide_model)
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

    #### Run forever
    poses     = torch.zeros(1,8,3,4)
    predposes = torch.zeros(1,8,3,4)
    predmasks = torch.zeros(1,8,240,320)
    config    = torch.zeros(1,7)
    ptcloud   = torch.zeros(1,3,240,320)
    while True:
        # Send to pangolin
        pangolin.update_viz(poses[0].numpy(), predposes[0].numpy(), predmasks[0].numpy(),
                            config[0].numpy(), ptcloud[0].numpy())

        # Run through net
        if args.use_jt_angles:
            inp = [util.to_var(ptcloud.type(deftype)), util.to_var(config.type(deftype))]
        else:
            inp = util.to_var(ptcloud.type(deftype))
        if args.use_gt_masks:  # GT masks are provided!
            predposes_n    = posemaskpredfn(inp)
        else:
            predposes_n, predmasks_n = posemaskpredfn(inp, train_iter=num_train_iter)
            predmasks.copy_(predmasks_n.data.cpu())
        predposes.copy_(predposes_n.data.cpu())

################ RUN MAIN
if __name__ == '__main__':
    main()