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
import torch.nn.functional as F
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
import configargparse

# Saved pose checkpoint
parser = configargparse.ArgumentParser(description='SE3-Pose-Nets Training')
parser.add_argument('--pose-resume', default='', type=str, metavar='PATH', required=True,
                    help='path to pre-trained pose model checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH', required=True,
                    help='path to saving pose data (default: none)')

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

    # Load pose model from a pre-trained checkpoint
    pargs = parser.parse_args()
    print("=> loading pose model checkpoint '{}'".format(pargs.pose_resume))
    pose_checkpoint = torch.load(pargs.pose_resume)
    num_train_iter = pose_checkpoint['train_iter']
    args = pose_checkpoint['args']
    print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
          .format(pargs.pose_resume, pose_checkpoint['epoch'], num_train_iter))
    best_pm_loss = pose_checkpoint['best_loss'] if 'best_loss' in pose_checkpoint else float("inf")
    best_pm_epoch = pose_checkpoint['best_epoch'] if 'best_epoch' in pose_checkpoint else 0
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_pm_loss, best_pm_epoch))

    ### Create save directory and start tensorboard logger
    util.create_dir(pargs.save_dir)  # Create directory
    now = time.strftime("%c")
    tblogger = util.TBLogger(pargs.save_dir + '/logs/' + now)  # Start tensorboard logger

    # Create logfile to save prints
    logfile = open(pargs.save_dir + '/logs/' + now + '/logfile.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

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
                                                 compute_bwdflows=args.use_gt_masks,
                                                 #num_tracker=args.num_tracker,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows,
                                                 noise_func=noise_func,
                                                 load_color=load_color,
                                                 compute_normals=(args.normal_wt > 0),
                                                 maxdepthdiff=args.normal_max_depth_diff,
                                                 bismooth_depths=args.bilateral_depth_smoothing,
                                                 bismooth_width=args.bilateral_window_width,
                                                 bismooth_std=args.bilateral_depth_std,
                                                 supervised_seg_loss=(args.seg_wt > 0)) # Need BWD flows / masks if using GT masks
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # Create a data-collater for combining the samples of the data into batches along with some post-processing
    train_sampler = torch.utils.data.dataloader.SequentialSampler(train_dataset)  # Run sequentially along the test dataset
    val_sampler   = torch.utils.data.dataloader.SequentialSampler(val_dataset)  # Run sequentially along the test dataset
    test_sampler  = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset

    ########################
    ############ Load models & optimization stuff

    assert not args.use_full_jt_angles, "Can only use as many jt angles as the control dimension"
    print('Using state of controllable joints')
    args.num_state_net = args.num_ctrl # Use only the jt angles of the controllable joints

    print('Using multi-step Flow-Model')
    if args.se2_data:
        print('Using the smaller multi-step SE2-Pose-Model')
    else:
        print('Using multi-step SE3-Pose-Model')

    ### Load the model
    #####
    num_input_channels = 3 # Num input channels
    if args.use_xyzrgb:
        num_input_channels = 6
    elif args.use_xyzhue:
        num_input_channels = 4 # Use only hue as input
    modelfn = ctrlnets.MultiStepSE3NoTransModel
    posemodel = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                        se3_type=args.se3_type, delta_pivot=args.delta_pivot, use_kinchain=False,
                        input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                        init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                        use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                        sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv, decomp_model=args.decomp_model,
                        use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                        wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                        use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state_net,
                        full_res=args.full_res, noise_stop_iter=args.noise_stop_iter) # noise_stop_iter not available for SE2 models
    if args.cuda:
        posemodel.cuda() # Convert to CUDA if enabled
    try:
        posemodel.load_state_dict(pose_checkpoint['state_dict']) # BWDs compatibility (TODO: remove)
    except:
        posemodel.load_state_dict(pose_checkpoint['model_state_dict'])

    ##########
    # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
    train_loader = util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                   collate_fn=train_dataset.collate_batch,
                                   sampler=train_sampler)
    train_stats = iterate(train_loader, posemodel, 'train')
    del train_loader
    print('Saving train stats to {}'.format(pargs.save_dir + "/transmodeldata_train.tar.gz"))
    torch.save(train_stats, pargs.save_dir + "/transmodeldata_train.tar.gz")

    val_loader   = util.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                   collate_fn=val_dataset.collate_batch,
                                   sampler=val_sampler)
    val_stats = iterate(val_loader, posemodel, 'val')
    del val_loader
    print('Saving val stats to {}'.format(pargs.save_dir + "/transmodeldata_val.tar.gz"))
    torch.save(val_stats, pargs.save_dir + "/transmodeldata_val.tar.gz")

    test_loader  = util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                   collate_fn=test_dataset.collate_batch,
                                   sampler=test_sampler)
    test_stats = iterate(test_loader, posemodel, 'test')
    del test_loader
    print('Saving test stats to {}'.format(pargs.save_dir + "/transmodeldata_test.tar.gz"))
    torch.save(test_stats, pargs.save_dir + "/transmodeldata_test.tar.gz")

    ########
    # print('Saving stats to {}'.format(pargs.save_dir + "/transmodeldata.tar.gz"))
    # torch.save({'train': train_stats, 'test': test_stats, 'val': val_stats},
    #            pargs.save_dir + "/transmodeldata.tar.gz")

def iterate(data_loader, posemodel, mode):
    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Save all stats into a namespace
    stats = argparse.Namespace()
    stats.gtposes_1, stats.gtposes_2, stats.predposes_1, stats.predposes_2, stats.ctrls_1 = [], [], [], [], []

    # Switch model modes
    posemodel.eval()

    # Run an epoch
    print('========== Mode: {}, Num iters: {} =========='.format(mode, len(data_loader)))
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'  # Default tensor type
    # Start timer
    start = time.time()
    for i, sample in enumerate(data_loader):
        # ============ Load data ============#
        if sample is None:
            print("All poses in the batch are NaNs. Discarding batch....")
            continue

        # Get a sample
        #j, sample = data_loader.next()

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts   = util.to_var(sample['points'].type(deftype), volatile=True)

        # Get XYZRGB input
        if args.use_xyzrgb:
            rgb = util.to_var(sample['rgbs'].type(deftype) / 255.0, requires_grad=False)  # Normalize RGB to 0-1
            netinput = torch.cat([pts, rgb], 2)  # Concat along channels dimension
        elif args.use_xyzhue:
            hue = util.to_var(sample['rgbs'].narrow(2, 0, 1).type(deftype) / 179.0,
                              requires_grad=False)  # Normalize Hue to 0-1 (Opencv has hue from 0-179)
            netinput = torch.cat([pts, hue], 2)  # Concat along channels dimension
        else:
            netinput = pts  # XYZ

        # Get jt angles
        if args.box_data:
            jtangles = util.to_var(sample['states'].type(deftype), requires_grad=False)
        else:
            jtangles = util.to_var(sample['actctrlconfigs'].type(deftype),
                                   requires_grad=False)  # [:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        ### Run a FWD pass through the network (multi-step)
        # Predict the poses and masks
        ### TODO: Make it more general with pivots etc
        predpose_1 = posemodel.forward_only_pose([netinput[:,0], jtangles[:,0]])
        predpose_2 = posemodel.forward_only_pose([netinput[:,1], jtangles[:,1]])
        stats.predposes_1.append(predpose_1.data.clone().view(-1,args.num_se3,3,4).cpu())
        stats.predposes_2.append(predpose_2.data.clone().view(-1,args.num_se3,3,4).cpu())
        stats.gtposes_1.append(sample['poses'][:,0].clone().cpu())
        stats.gtposes_2.append(sample['poses'][:,1].clone().cpu())
        stats.ctrls_1.append(sample['controls'][:,0].clone().cpu())

        # Measure data loading time
        fwd_time.update(time.time() - start)

        ###
        if (i % 100) == 0:
            print('Mode: {}, Iter: {: 5}/{}'.format(mode, i, len(data_loader)))
            print('\tTime => Data: {data.val:.3f} ({data.avg:.3f}), '
                  'Fwd: {fwd.val:.3f} ({fwd.avg:.3f})'.format(
                data=data_time, fwd=fwd_time))

        # Reset timer
        start = time.time()

    ###
    return stats

################ RUN MAIN
if __name__ == '__main__':
    main()