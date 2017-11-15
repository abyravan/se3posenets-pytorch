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
                                                 compute_bwdflows=args.use_gt_masks,
                                                 #num_tracker=args.num_tracker,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows,
                                                 noise_func=noise_func,
                                                 load_color=args.use_xyzrgb) # Need BWD flows / masks if using GT masks
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
    num_train_iter = 0
    num_input_channels = 6 if args.use_xyzrgb else 3 # Num input channels
    if args.use_gt_masks:
        print('Using GT masks. Model predicts only poses & delta-poses')
        assert not args.use_gt_poses, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.MultiStepSE2OnlyPoseModel if args.se2_data else ctrlnets.MultiStepSE3OnlyPoseModel
    elif args.use_gt_poses:
        print('Using GT poses & delta poses. Model predicts only masks')
        assert not args.use_gt_masks, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.MultiStepSE2OnlyMaskModel if args.se2_data else ctrlnets.MultiStepSE3OnlyMaskModel
    else:
        modelfn = se2nets.MultiStepSE2PoseModel if args.se2_data else ctrlnets.MultiStepSE3PoseModel
    model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                    se3_type=args.se3_type, delta_pivot=args.delta_pivot, use_kinchain=False,
                    input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                    init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                    use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                    sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv, decomp_model=args.decomp_model,
                    use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                    wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                    use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state_net,
                    full_res=args.full_res)
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    ### Load optimizer
    optimizer = load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        # TODO: Save path to TB log dir, save new log there again
        # TODO: Reuse options in args (see what all to use and what not)
        # TODO: Use same num train iters as the saved checkpoint
        # TODO: Print some stats on the training so far, reset best validation loss, best epoch etc
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint       = torch.load(args.resume)
            loadargs         = checkpoint['args']
            args.start_epoch = checkpoint['epoch']
            if args.reset_train_iter:
                num_train_iter   = 0 # Reset to 0
            else:
                num_train_iter   = checkpoint['train_iter']
            try:
                model.load_state_dict(checkpoint['state_dict']) # BWDs compatibility (TODO: remove)
            except:
                model.load_state_dict(checkpoint['model_state_dict'])
            assert (loadargs.optimization == args.optimization), "Optimizer in saved checkpoint ({}) does not match current argument ({})".format(
                    loadargs.optimization, args.optimization)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
                  .format(args.resume, checkpoint['epoch'], num_train_iter))
            best_val_loss = checkpoint['best_loss']
            best_epoch = checkpoint['best_epoch'] if hasattr(checkpoint, 'best_epoch') else 0
            print('==== Best validation loss: {} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                                 best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_val_loss, best_epoch = float("inf"), 0

    ########################
    ############ Test (don't create the data loader unless needed, creates 4 extra threads)
    if args.evaluate:
        # Delete train and val loaders
        #del train_loader, val_loader

        # TODO: Move this to before the train/val loader creation??
        print('==== Evaluating pre-trained network on test data ===')
        test_stats = iterate(test_loader, model, tblogger, len(test_loader), mode='test')

        # Save final test error
        save_checkpoint({
            'args': args,
            'test_stats': {'stats': test_stats,
                           'niters': test_loader.niters, 'nruns': test_loader.nruns,
                           'totaliters': test_loader.iteration_count(),
                           'ids': test_stats.data_ids,
                           },
        }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

        # Close log file & return
        logfile.close()
        return

    ########################
    ############ Train / Validate
    args.imgdisp_freq = 5 * args.disp_freq # Tensorboard log frequency for the image data
    train_ids, val_ids = [], []
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs)

        # Train for one epoch
        train_stats = iterate(train_loader, model, tblogger, args.train_ipe,
                           mode='train', optimizer=optimizer, epoch=epoch+1)
        train_ids += train_stats.data_ids

        # Evaluate on validation set
        val_stats = iterate(val_loader, model, tblogger, args.val_ipe,
                            mode='val', epoch=epoch+1)
        val_ids += val_stats.data_ids

        # Find best loss
        val_loss = val_stats.loss
        is_best       = (val_loss.avg < best_val_loss)
        prev_best_loss  = best_val_loss
        prev_best_epoch = best_epoch
        if is_best:
            best_val_loss = val_loss.avg
            best_epoch    = epoch+1
            print('==== Epoch: {}, Improved on previous best loss ({}) from epoch {}. Current: {} ===='.format(
                                    epoch+1, prev_best_loss, prev_best_epoch, val_loss.avg))
        else:
            print('==== Epoch: {}, Did not improve on best loss ({}) from epoch {}. Current: {} ===='.format(
                epoch + 1, prev_best_loss, prev_best_epoch, val_loss.avg))

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch+1,
            'args' : args,
            'best_loss'  : best_val_loss,
            'best_epoch' : best_epoch,
            'train_stats': {'stats': train_stats,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count(),
                            'ids': train_ids,
                            },
            'val_stats'  : {'stats': val_stats,
                            'niters': val_loader.niters, 'nruns': val_loader.nruns,
                            'totaliters': val_loader.iteration_count(),
                            'ids': val_ids,
                            },
            'train_iter' : num_train_iter,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, is_best, savedir=args.save_dir, filename='checkpoint.pth.tar') #_{}.pth.tar'.format(epoch+1))
        print('\n')

    # Delete train and val data loaders
    del train_loader, val_loader

    # Load best model for testing (not latest one)
    print("=> loading best model from '{}'".format(args.save_dir + "/model_best.pth.tar"))
    checkpoint = torch.load(args.save_dir + "/model_best.pth.tar")
    num_train_iter = checkpoint['train_iter']
    try:
        model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded best checkpoint (epoch {}, train iter {})"
          .format(checkpoint['epoch'], num_train_iter))
    best_epoch = checkpoint['best_epoch'] if hasattr(checkpoint, 'best_epoch') else 0
    print('==== Best validation loss: {} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))

    # Do final testing (if not asked to evaluate)
    # (don't create the data loader unless needed, creates 4 extra threads)
    print('==== Evaluating trained network on test data ====')
    args.imgdisp_freq = 10 * args.disp_freq # Tensorboard log frequency for the image data
    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, sampler=sampler, pin_memory=args.use_pin_memory,
                                    collate_fn=test_dataset.collate_batch))
    test_stats = iterate(test_loader, model, tblogger, len(test_loader),
                         mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'test_stats': {'stats': test_stats,
                       'niters': test_loader.niters, 'nruns': test_loader.nruns,
                       'totaliters': test_loader.iteration_count(),
                       'ids': test_stats.data_ids,
                       },
    }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

    # Close log file
    logfile.close()

################# HELPER FUNCTIONS

### Main iterate function (train/test/val)
def iterate(data_loader, model, tblogger, num_iters,
            mode='test', optimizer=None, epoch=0):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Save all stats into a namespace
    stats = argparse.Namespace()
    stats.loss, stats.ptloss, stats.consisloss  = AverageMeter(), AverageMeter(), AverageMeter()
    stats.dissimposeloss, stats.dissimdeltaloss = AverageMeter(), AverageMeter()
    stats.flowerr_sum, stats.flowerr_avg        = AverageMeter(), AverageMeter()
    stats.motionerr_sum, stats.motionerr_avg    = AverageMeter(), AverageMeter()
    stats.stillerr_sum, stats.stillerr_avg      = AverageMeter(), AverageMeter()
    stats.consiserr                             = AverageMeter()
    stats.data_ids = []
    if mode == 'test':
        # Save the flow errors and poses if in "testing" mode
        stats.motion_err, stats.motion_npt, stats.still_err, stats.still_npt = [], [], [], []
        stats.predposes, stats.predtransposes, stats.preddeltas, stats.ctrls = [], [], [], []
        stats.poses = []
        # stats.predmasks, stats.masks = [], []
        # stats.gtflows, stats.predflows = [], []
        # stats.pts = []

    # Switch model modes
    train = True if (mode == 'train') else False
    if train:
        assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
        model.train()
    else:
        assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}"+mode
        model.eval()

    # Create a closure to get the outputs of the delta-se3 prediction layers
    predictions = {}
    def get_output(name):
        def hook(self, input, result):
            predictions[name] = result
        return hook
    model.transitionmodel.deltase3decoder.register_forward_hook(get_output('deltase3'))

    # Point predictor
    # NOTE: The prediction outputs of both layers are the same if mask normalization is used, if sigmoid the outputs are different
    # NOTE: Gradients are same for pts & tfms if mask normalization is used, always different for the masks
    ptpredlayer = se3nn.NTfm3D

    # Type of loss (mixture of experts = wt sharpening or sigmoid)
    mex_loss = True

    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(
        mode, epoch, num_iters))
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    pt_wt, consis_wt = args.pt_wt * args.loss_scale, args.consis_wt * args.loss_scale
    identfm = util.to_var(torch.eye(4).view(1,1,4,4).expand(1,args.num_se3-1,4,4).narrow(2,0,3).type(deftype), requires_grad=False)
    for i in xrange(num_iters):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        j, sample = data_loader.next()
        stats.data_ids.append(sample['id'].clone())

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts      = util.to_var(sample['points'].type(deftype), requires_grad=train, volatile=not train)
        ctrls    = util.to_var(sample['controls'].type(deftype), requires_grad=train)
        fwdflows = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        fwdvis   = util.to_var(sample['fwdvisibilities'].type(deftype), requires_grad=False)
        tarpts   = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        tarpts.data.add_(pts.data.narrow(1,0,1).expand_as(tarpts.data)) # Add "k"-step flows to the initial point cloud

        # Get XYZRGB input
        if args.use_xyzrgb:
            rgb = util.to_var(sample['rgbs'].type(deftype)/255.0, requires_grad=train) # Normalize RGB to 0-1
            netinput = torch.cat([pts, rgb], 2) # Concat along channels dimension
        else:
            netinput = pts # XYZ

        # Get jt angles
        #if args.use_full_jt_angles:
        #    jtangles = util.to_var(sample['actconfigs'].type(deftype), requires_grad=train)
        #else:
        if args.box_data:
            jtangles = util.to_var(sample['states'].type(deftype), requires_grad=train)
        else:
            jtangles = util.to_var(sample['actctrlconfigs'].type(deftype), requires_grad=train) #[:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        ### Run a FWD pass through the network (multi-step)
        # Predict the poses and masks
        poses, initmask = [], None
        masks, pivots = [], []
        maskcenters, posecenters = [], []
        for k in xrange(pts.size(1)):
            if ((args.delta_pivot == '') or (args.delta_pivot == 'pred')) and args.pose_center == 'pred':
                # Predict the pose and mask at time t = 0
                # For all subsequent timesteps, predict only the poses
                if(k == 0):
                    if args.use_gt_masks:
                        p = model.forward_only_pose([netinput[:,k], jtangles[:,k]]) # We can only predict poses
                        initmask = util.to_var(sample['masks'][:,0].type(deftype).clone(), requires_grad=False) # Use GT masks
                    elif args.use_gt_poses:
                        p = util.to_var(sample['poses'][:,k].type(deftype).clone(), requires_grad=False)  # Use GT poses
                        initmask = model.forward_only_mask(netinput[:,k], train_iter=num_train_iter)  # Predict masks
                    else:
                        p, initmask = model.forward_pose_mask([netinput[:,k], jtangles[:,k]], train_iter=num_train_iter)
                else:
                    if args.use_gt_poses:
                        p = util.to_var(sample['poses'][:,k].type(deftype).clone(), requires_grad=False)  # Use GT poses
                    else:
                        p = model.forward_only_pose([netinput[:,k], jtangles[:,k]])
                poses.append(p)
            else:
                # Predict the poses and masks for all timesteps
                if args.use_gt_masks:
                    p = model.forward_only_pose([netinput[:,k], jtangles[:,k]])  # We can only predict poses
                    m = util.to_var(sample['masks'][:,k].type(deftype).clone(), requires_grad=False)  # Use GT masks
                elif args.use_gt_poses:
                    p = util.to_var(sample['poses'][:,k].type(deftype).clone(), requires_grad=False)  # Use GT poses
                    m = model.forward_only_mask(netinput[:,k], train_iter=num_train_iter) # Predict masks
                else:
                    p, m = model.forward_pose_mask([netinput[:,k], jtangles[:,k]], train_iter=num_train_iter) # TODO: Mask computations only needed for "maskmean" & "maskmeannograd"
                masks.append(m)
                # Update poses if there is a center option provided
                p, pc, mc = ctrlnets.update_pose_centers(pts[:,k], m, p, args.pose_center)
                poses.append(p)
                if pc is not None:
                    posecenters.append(pc)
                    maskcenters.append(mc)
                # Compute pivots
                pivots.append(ctrlnets.compute_pivots(pts[:,k], m, p, args.delta_pivot))
                # Get initial mask
                if (k == 0):
                    initmask = masks[0]

        ## Make next-pose predictions & corresponding 3D point predictions using the transition model
        ## We use pose_0 and [ctrl_0, ctrl_1, .., ctrl_(T-1)] to make the predictions
        ## NOTE: Here, each step is independent of the previous and we do not want to backprop across the chains.
        ## NOTE: We also need to make copies of the deltas and compose them over time. Again, the key is that we do not
        ## have any chains between the deltas - for delta_k, delta_0->k-1 is a fixed input, not a variable
        deltaposes, transposes, compdeltaposes = [], [], []
        for k in xrange(args.seq_len):
            # Get current pose (backwards graph only connects to encoder through poses[0])
            # No gradient feedback within multiple steps of the transition model
            if (k == 0):
                pose = poses[0] # Use initial pose (NOTE: We want to backprop to the pose model here)
            else:
                pose = util.to_var(transposes[k-1].data.clone(), requires_grad=False) # Use previous predicted pose (NOTE: This is a copy with the graph cut)

            # Predict next pose based on curr pose, control
            delta, trans = model.forward_next_pose(pose, ctrls[:,k], jtangles[:,k],
                                                   pivots[k] if (len(pivots) > 0) else None)
            deltaposes.append(delta)
            transposes.append(trans)

            # Compose the deltas over time (T4 = T4 * T3^-1 * T3 * T2^-1 * T2 * T1^-1 * T1 = delta_4 * delta_3 * delta_2 * T1
            # NOTE: This is only used if backprop-only-first-delta is set. Otherwise we use the deltas directly
            if args.backprop_only_first_delta:
                if (k == 0):
                    compdeltaposes.append(delta) # This keeps the graph intact
                else:
                    prevcompdelta = util.to_var(compdeltaposes[k-1].data.clone(), requires_grad=False) # Cut graph here, no gradient feedback in transition model
                    compdeltaposes.append(se3nn.ComposeRtPair()(delta, prevcompdelta)) # delta_full = delta_curr * delta_prev

        # Now compute the losses across the sequence
        # We use point loss in the FWD dirn and Consistency loss between poses
        predpts, ptloss, consisloss, loss = [], torch.zeros(args.seq_len), torch.zeros(args.seq_len), 0
        dissimposeloss, dissimdeltaloss = torch.zeros(args.seq_len), torch.zeros(args.seq_len)
        for k in xrange(args.seq_len):
            ### Make the 3D point predictions and set up loss
            # Separate between using the full gradient & gradient only for the first delta pose
            if args.backprop_only_first_delta:
                # Predict transformed 3D points
                # We back-propagate only to the first predicted delta, so we break the graph between the later deltas and the predicted 3D points
                compdelta = compdeltaposes[k] if (k == 0) else util.to_var(compdeltaposes[k].data.clone(), requires_grad=False)
                nextpts = ptpredlayer()(pts[:,0], initmask, compdelta)

                # Setup inputs & targets for loss
                # NOTE: These losses are correct for the masks, but for any delta other than the first, the errors are
                # not correct since we will have a credit assignment problem where each step's delta has to optimize for the complete errors
                # So we only backprop to the first delta
                inputs = nextpts - pts[:,0]
                targets = fwdflows[:,k]
            else:
                # Predict transformed 3D points
                # We do not want to backpropagate across the chain of predicted points so we break the graph here
                currpts = pts[:,0] if (k == 0) else util.to_var(predpts[k-1].data.clone(), requires_grad=False)
                nextpts = ptpredlayer()(currpts, initmask, deltaposes[k]) # We do want to backpropagate to all the deltas in this case

                # Setup inputs & targets for loss
                # For each step, we only look at the target flow for that step (how much do those points move based on that control alone)
                # and compare that against the predicted flows for that step alone!
                # TODO: This still is wrong as "currpts" will be wrong if at the previous timestep the deltas were incorrect.
                # TODO: So the gradients will not be totally correct since even though we look at delta-flow errors here, the points which
                # TODO: are transformed to compute those errors are not correct which can lead to incorrect motion predictions
                inputs = nextpts - currpts  # Delta flow for that step (note that gradients only go to the mask & deltas)
                targets = fwdflows[:,k] - (0 if (k == 0) else fwdflows[:,k-1])  # Flow for those points in that step alone!
            predpts.append(nextpts) # Save predicted pts

            ### 3D loss
            # If motion-normalized loss, pass in GT flows
            if args.motion_norm_loss:
                motion  = targets # Use either delta-flows or full-flows
                currptloss = pt_wt * ctrlnets.MotionNormalizedLoss3D(inputs, targets, motion=motion,
                                                                     loss_type=args.loss_type, wts=fwdvis[:, k])
            else:
                currptloss = pt_wt * ctrlnets.Loss3D(inputs, targets, loss_type=args.loss_type, wts=fwdvis[:, k])

            ### Consistency loss (between t & t+1)
            # Poses from encoder @ t & @ t+1 should be separated by delta from t->t+1
            # NOTE: For the consistency loss, the loss is only backpropagated to the encoder poses, not to the deltas
            delta = util.to_var(deltaposes[k].data.clone(), requires_grad=False)  # Break the graph here
            nextpose_trans = se3nn.ComposeRtPair()(delta, poses[k])
            if args.consis_rt_loss:
                currconsisloss = consis_wt * ctrlnets.PoseC(nextpose_trans, poses[k+1])
            else:
                currconsisloss = consis_wt * ctrlnets.BiMSELoss(nextpose_trans, poses[k+1])

            # Add a loss for pose dis-similarity & delta dis-similarity
            dissimpose_wt, dissimdelta_wt = args.pose_dissim_wt * args.loss_scale, args.delta_dissim_wt * args.loss_scale
            currdissimposeloss  = dissimpose_wt * ctrlnets.DisSimilarityLoss(poses[k][:,1:],
                                                                             poses[k+1][:,1:],
                                                                             size_average=True)  # Enforce dis-similarity in pose space
            currdissimdeltaloss = dissimdelta_wt * ctrlnets.DisSimilarityLoss(deltaposes[k][:,1:],
                                                                      identfm.expand_as(deltaposes[k][:,1:]),
                                                                      size_average=True) # Change in pose > 0

            # Append to total loss
            loss += currptloss + currconsisloss # + currdissimposeloss + currdissimdeltaloss
            ptloss[k]     = currptloss.data[0]
            consisloss[k] = currconsisloss.data[0]
            dissimposeloss[k]  = currdissimposeloss.data[0]
            dissimdeltaloss[k] = currdissimdeltaloss.data[0]

        # Update stats
        stats.ptloss.update(ptloss)
        stats.consisloss.update(consisloss)
        stats.loss.update(loss.data[0])
        stats.dissimposeloss.update(dissimposeloss)
        stats.dissimdeltaloss.update(dissimdeltaloss)

        # Measure FWD time
        fwd_time.update(time.time() - start)

        # ============ Gradient backpass + Optimizer step ============#
        # Compute gradient and do optimizer update step (if in training mode)
        if (train):
            # Start timer
            start = time.time()

            # Backward pass & optimize
            optimizer.zero_grad() # Zero gradients
            loss.backward()       # Compute gradients - BWD pass
            optimizer.step()      # Run update step

            # Increment number of training iterations by 1
            num_train_iter += 1

            # Measure BWD time
            bwd_time.update(time.time() - start)

        # ============ Visualization ============#
        # Start timer
        start = time.time()

        # Compute flow predictions and errors
        # NOTE: I'm using CUDA here to speed up computation by ~4x
        predflows = torch.cat([(x.data - pts.data[:,0]).unsqueeze(1) for x in predpts], 1)
        flows = fwdflows.data
        if args.use_only_da_for_flows:
            # If using only DA then pts that are not visible will not have GT flows, so we shouldn't take them into
            # account when computing the flow errors
            flowerr_sum, flowerr_avg, \
                motionerr_sum, motionerr_avg,\
                stillerr_sum, stillerr_avg,\
                motion_err, motion_npt,\
                still_err, still_npt         = compute_masked_flow_errors(predflows * fwdvis, flows) # Zero out flows for non-visible points
        else:
            flowerr_sum, flowerr_avg, \
                motionerr_sum, motionerr_avg, \
                stillerr_sum, stillerr_avg, \
                motion_err, motion_npt, \
                still_err, still_npt         = compute_masked_flow_errors(predflows, flows)

        # Update stats
        stats.flowerr_sum.update(flowerr_sum); stats.flowerr_avg.update(flowerr_avg)
        stats.motionerr_sum.update(motionerr_sum); stats.motionerr_avg.update(motionerr_avg)
        stats.stillerr_sum.update(stillerr_sum); stats.stillerr_avg.update(stillerr_avg)
        if mode == 'test':
            stats.motion_err.append(motion_err); stats.motion_npt.append(motion_npt)
            stats.still_err.append(still_err); stats.still_npt.append(still_npt)

        # Save poses if in test mode
        if mode == 'test':
            stats.predposes.append([x.data.cpu().float() for x in poses])
            stats.predtransposes.append([x.data.cpu().float() for x in transposes])
            stats.preddeltas.append([x.data.cpu().float() for x in deltaposes])
            stats.ctrls.append(ctrls.data.cpu().float())
            stats.poses.append(sample['poses'])
            # stats.predmasks.append(initmask.data.cpu().float())
            # stats.masks.append(sample['masks'][:,0])
            # stats.predflows.append(predflows.cpu())
            # stats.gtflows.append(flows.cpu())
            # stats.pts.append(sample['points'][:,0])

        # Compute flow error per mask (if asked to)
        #if args.disp_err_per_mask:
        #    flowloss_mask_sum_fwd, flowloss_mask_avg_fwd, _, _ = compute_flow_errors_per_mask(predflows.data,
        #                                                                                      flows.data,
        #                                                                                      sample['gtmasks'])

        ### Pose consistency error
        # Compute consistency error for display
        consiserror, consiserrormax = torch.zeros(args.seq_len), torch.zeros(args.seq_len)
        for k in xrange(args.seq_len):
            consiserrormax[k] = (poses[k+1].data - transposes[k].data).abs().max()
            consiserror[k] = ctrlnets.BiAbsLoss(poses[k+1].data, transposes[k].data)
        stats.consiserr.update(consiserror)

        # Display/Print frequency
        bsz = pts.size(0)
        if i % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                        samplecurr=j+1, sampletotal=len(data_loader),
                        stats=stats, bsz=bsz)

            ### Print stuff if we have weight sharpening enabled
            if args.use_wt_sharpening and not args.use_gt_masks:
                try:
                    noise_std, pow = model.posemaskmodel.compute_wt_sharpening_stats(train_iter=num_train_iter)
                except:
                    noise_std, pow = model.maskmodel.compute_wt_sharpening_stats(train_iter=num_train_iter)
                print('\tWeight sharpening => Num training iters: {}, Noise std: {:.4f}, Power: {:.3f}'.format(
                    num_train_iter, noise_std, pow))

            ### Print time taken
            print('\tTime => Data: {data.val:.3f} ({data.avg:.3f}), '
                        'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
                        'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
                        'Viz: {viz.val:.3f} ({viz.avg:.3f})'.format(
                    data=data_time, fwd=fwd_time, bwd=bwd_time, viz=viz_time))

            ### TensorBoard logging
            # (1) Log the scalar values
            iterct = data_loader.iteration_count() # Get total number of iterations so far
            info = {
                mode+'-loss': loss.data[0],
                mode+'-pt3dloss': ptloss.sum(),
                mode+'-consisloss': consisloss.sum(),
                mode+'-dissimposeloss': dissimposeloss.sum(),
                mode+'-dissimdeltaloss': dissimdeltaloss.sum(),
                mode+'-consiserr': consiserror.sum(),
                mode+'-consiserrmax': consiserrormax.sum(),
                mode+'-flowerrsum': flowerr_sum.sum()/bsz,
                mode+'-flowerravg': flowerr_avg.sum()/bsz,
                mode+'-motionerrsum': motionerr_sum.sum()/bsz,
                mode+'-motionerravg': motionerr_avg.sum()/bsz,
                mode+'-stillerrsum': stillerr_sum.sum() / bsz,
                mode+'-stillerravg': stillerr_avg.sum() / bsz,
            }
            for tag, value in info.items():
                tblogger.scalar_summary(tag, value, iterct)

            # (2) Log images & print predicted SE3s
            # TODO: Numpy or matplotlib
            if i % args.imgdisp_freq == 0:

                ## Log the images (at a lower rate for now)
                id = random.randint(0, sample['points'].size(0)-1)

                # Render the predicted and GT poses onto the depth
                depths = []
                for k in xrange(args.seq_len+1):
                    gtpose    = sample['poses'][id, k]
                    predpose  = poses[k].data[id].cpu().float()
                    predposet = transposes[k-1].data[id].cpu().float() if (k > 0) else None
                    gtdepth   = normalize_img(sample['points'][id,k,2:].expand(3,args.img_ht,args.img_wd).permute(1,2,0), min=0, max=3)
                    for n in xrange(args.num_se3):
                        # Pose_1 (GT/Pred)
                        if n < gtpose.size(0):
                            util.draw_3d_frame(gtdepth, gtpose[n], [0,0,1], args.cam_intrinsics[0], pixlength=15.0) # GT pose: Blue
                        util.draw_3d_frame(gtdepth, predpose[n], [0,1,0], args.cam_intrinsics[0], pixlength=15.0) # Pred pose: Green
                        if predposet is not None:
                            util.draw_3d_frame(gtdepth, predposet[n], [1,0,0], args.cam_intrinsics[0], pixlength=15.0)  # Transition model pred pose: Red
                    depths.append(gtdepth)
                depthdisp = torch.cat(depths, 1).permute(2,0,1) # Concatenate along columns (3 x 240 x 320*seq_len+1 image)

                # Concat the flows, depths and masks into one tensor
                flowdisp  = torchvision.utils.make_grid(torch.cat([flows.narrow(0,id,1),
                                                                   predflows.narrow(0,id,1)], 0).cpu().view(-1, 3, args.img_ht, args.img_wd),
                                                        nrow=args.seq_len, normalize=True, range=(-0.01, 0.01))
                #depthdisp = torchvision.utils.make_grid(sample['points'][id].narrow(1,2,1), normalize=True, range=(0.0,3.0))
                maskdisp  = torchvision.utils.make_grid(torch.cat([initmask.data.narrow(0,id,1)], 0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                                                        nrow=args.num_se3, normalize=True, range=(0,1))

                # Display RGB
                if args.use_xyzrgb:
                    rgbdisp = torchvision.utils.make_grid(sample['rgbs'][id].float().view(-1, 3, args.img_ht, args.img_wd),
                                                            nrow=args.seq_len, normalize=True, range=(0.0,255.0))

                # Show as an image summary
                info = { mode+'-depths': util.to_np(depthdisp.unsqueeze(0)),
                         mode+'-flows' : util.to_np(flowdisp.unsqueeze(0)),
                         mode+'-masks' : util.to_np(maskdisp.narrow(0,0,1))
                }
                if args.use_xyzrgb:
                    info[mode+'-rgbs'] = util.to_np(rgbdisp.unsqueeze(0)) # Optional RGB
                for tag, images in info.items():
                    tblogger.image_summary(tag, images, iterct)

                ## Print the predicted delta-SE3s
                deltase3s = predictions['deltase3'].data[id].view(args.num_se3, -1).cpu()
                if len(pivots) > 0:
                    deltase3s = torch.cat([deltase3s, pivots[-1].data[id].view(args.num_se3,-1).cpu()], 1)
                print('\tPredicted delta-SE3s @ t=2:', deltase3s)

                ## Details on the centers
                if len(posecenters) > 0:
                    centers = torch.cat([maskcenters[-1].data[id].cpu(), posecenters[-1].data[id].cpu(), poses[-1].data[id,:,:,3].cpu()], 1)
                    print('\tMaskCenters | PoseCenters (Init) | PoseCenters (Final)', centers)

                ## Print the predicted mask values
                print('\tPredicted mask stats:')
                for k in xrange(args.num_se3):
                    print('\tMax: {:.4f}, Min: {:.4f}, Mean: {:.4f}, Std: {:.4f}, Median: {:.4f}, Pred 1: {}'.format(
                        initmask.data[id,k].max(), initmask.data[id,k].min(), initmask.data[id,k].mean(),
                        initmask.data[id,k].std(), initmask.data[id,k].view(-1).cpu().float().median(),
                        (initmask.data[id,k] - 1).abs().le(1e-5).sum()))
                print('')

        # Measure viz time
        viz_time.update(time.time() - start)

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=num_iters, total=num_iters,
                samplecurr=data_loader.niters+1, sampletotal=len(data_loader),
                stats=stats)
    print('========================================================')

    # Return the loss & flow loss
    return stats

### Print statistics
def print_stats(mode, epoch, curr, total, samplecurr, sampletotal,
                stats, bsz=None):
    # Print loss
    bsz = args.batch_size if bsz is None else bsz
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], Batch size: {}, '
          'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, bsz, loss=stats.loss))

    # Print flow loss per timestep
    for k in xrange(args.seq_len):
        print('\tStep: {}, Pt: {:.3f} ({:.3f}), Consis: {:.3f}/{:.4f} ({:.3f}/{:.4f}), '
              'Pose-Dissim: {:.3f} ({:.3f}), Delta-Dissim: {:.3f} ({:.3f}), '
              'Flow => Sum: {:.3f} ({:.3f}), Avg: {:.3f} ({:.3f}), '
              'Motion/Still => Sum: {:.3f}/{:.3f}, Avg: {:.3f}/{:.3f}'
            .format(
            1 + k * args.step_len,
            stats.ptloss.val[k], stats.ptloss.avg[k],
            stats.consisloss.val[k], stats.consisloss.avg[k],
            stats.consiserr.val[k], stats.consiserr.avg[k],
            stats.dissimposeloss.val[k], stats.dissimposeloss.avg[k],
            stats.dissimdeltaloss.val[k], stats.dissimdeltaloss.avg[k],
            stats.flowerr_sum.val[k] / bsz, stats.flowerr_sum.avg[k] / bsz,
            stats.flowerr_avg.val[k] / bsz, stats.flowerr_avg.avg[k] / bsz,
            stats.motionerr_sum.avg[k] / bsz, stats.stillerr_sum.avg[k] / bsz,
            stats.motionerr_avg.avg[k] / bsz, stats.stillerr_avg.avg[k] / bsz,
        ))

### Load optimizer
def load_optimizer(optim_type, parameters, lr=1e-3, momentum=0.9, weight_decay=1e-4):
    if optim_type == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr = lr, weight_decay= weight_decay)
    else:
        assert False, "Unknown optimizer type: " + optim_type
    return optimizer

### Save checkpoint
def save_checkpoint(state, is_best, savedir='.', filename='checkpoint.pth.tar'):
    savefile = savedir + '/' + filename
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, savedir + '/model_best.pth.tar')

### Compute flow errors for moving / non-moving pts (flows are size: B x S x 3 x H x W)
def compute_masked_flow_errors(predflows, gtflows):
    batch, seq = predflows.size(0), predflows.size(1) # B x S x 3 x H x W
    # Compute num pts not moving per mask
    # !!!!!!!!! > 1e-3 returns a ByteTensor and if u sum within byte tensors, the max value we can get is 255 !!!!!!!!!
    motionmask = (gtflows.abs().sum(2) > 1e-3).type_as(gtflows) # B x S x 1 x H x W
    err = (predflows - gtflows).mul_(1e2).pow(2).sum(2) # B x S x 1 x H x W

    # Compute errors for points that are supposed to move
    motion_err = (err * motionmask).view(batch, seq, -1).sum(2) # Errors for only those points that are supposed to move
    motion_npt = motionmask.view(batch, seq, -1).sum(2) # Num points that move (B x S)

    # Compute errors for points that are supposed to not move
    motionmask.eq_(0) # Mask out points that are not supposed to move
    still_err = (err * motionmask).view(batch, seq, -1).sum(2)  # Errors for non-moving points
    still_npt = motionmask.view(batch, seq, -1).sum(2)  # Num non-moving pts (B x S)

    # Bwds compatibility to old error
    full_err_avg  = (motion_err + still_err) / motion_npt
    full_err_avg[full_err_avg != full_err_avg] = 0  # Clear out any Nans
    full_err_avg[full_err_avg == np.inf] = 0  # Clear out any Infs
    full_err_sum, full_err_avg = (motion_err + still_err).sum(0), full_err_avg.sum(0) # S, S

    # Compute sum/avg stats
    motion_err_avg = (motion_err / motion_npt)
    motion_err_avg[motion_err_avg != motion_err_avg] = 0  # Clear out any Nans
    motion_err_avg[motion_err_avg == np.inf] = 0      # Clear out any Infs
    motion_err_sum, motion_err_avg = motion_err.sum(0), motion_err_avg.sum(0) # S, S

    # Compute sum/avg stats
    still_err_avg = (still_err / still_npt)
    still_err_avg[still_err_avg != still_err_avg] = 0  # Clear out any Nans
    still_err_avg[still_err_avg == np.inf] = 0  # Clear out any Infs
    still_err_sum, still_err_avg = still_err.sum(0), still_err_avg.sum(0)  # S, S

    # Return
    return full_err_sum.cpu().float(), full_err_avg.cpu().float(), \
           motion_err_sum.cpu().float(), motion_err_avg.cpu().float(), \
           still_err_sum.cpu().float(), still_err_avg.cpu().float(), \
           motion_err.cpu().float(), motion_npt.cpu().float(), \
           still_err.cpu().float(), still_npt.cpu().float()

### Normalize image
def normalize_img(img, min=-0.01, max=0.01):
    return (img - min) / (max - min)

### Adjust learning rate
def adjust_learning_rate(optimizer, epoch, decay_rate=0.1, decay_epochs=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

################ RUN MAIN
if __name__ == '__main__':
    main()
