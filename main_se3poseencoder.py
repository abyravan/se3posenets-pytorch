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
import util
from util import AverageMeter, Tee, DataEnumerator

## New layers
import se3
import se3posenets

#### Setup options
# Common
import argparse
import options
parser = options.setup_comon_options()

# Loss options
parser.add_argument('--pt-wt', default=1, type=float,
                    metavar='WT', help='Weight for the 3D point loss - only FWD direction (default: 1)')

# Mask consistency loss options
parser.add_argument('--mask-consis-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the mask consistency loss - only FWD direction (default: 0.01)')
parser.add_argument('--mask-consis-loss-type', default='mse', type=str,
                    metavar='STR', help='Type of loss to use for mask consistency errors, '
                                        '(default: mse | abs | kl | kllog)')
parser.add_argument('--pre-mask-consis', action='store_true', default=False,
                    help='Use the pre-sharpened activations for mask consistency loss (default: False)')

# Use SE3NN
parser.add_argument('--use-se3nn', action='store_true', default=False,
                    help='Use SE3NN SE3ToRt layer instead of the ones in se3.py (default: False)')

# Compose deltas for multi-step
parser.add_argument('--compose-deltas', action='store_true', default=False,
                    help='Compose delta SE3s over time (default: False)')

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
    elif args.use_xyzhue:
        print("Using XYZ-Hue input - 4 channels. Assumes registered depth/RGB")

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

        ### BAXTER DATA
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
    args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat', 'se3aar']), 'Unknown SE3 type: ' + args.se3_type

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
        args.loss_scale, args.pt_wt, args.consis_wt))

    # Weight sharpening stuff
    if args.use_wt_sharpening:
        print('Using weight sharpening to encourage binary mask prediction. Start iter: {}, Rate: {}, Noise stop iter: {}'.format(
            args.sharpen_start_iter, args.sharpen_rate, args.noise_stop_iter))

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

    if args.use_se3nn:
        print('Using SE3NNs SE3ToRt layer implementation')
    else:
        print('Using the SE3ToRt implementation in se3.py')

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

    ### Baxter dataset
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
                                                 mesh_ids = args.mesh_ids,
                                                 compute_bwdflows=args.use_gt_masks,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows,
                                                 noise_func=noise_func,
                                                 load_color=load_color) # Need BWD flows / masks if using GT masks
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

    #assert not args.use_full_jt_angles, "Can only use as many jt angles as the control dimension"
    print('Using state of controllable joints')
    args.num_state_net = args.num_ctrl # Use only the jt angles of the controllable joints
    if args.se2_data:
        assert(False)

    ### Load the model
    num_train_iter = 0
    num_input_channels = 3 # Num input channels
    if args.use_xyzrgb:
        num_input_channels = 6
    elif args.use_xyzhue:
        num_input_channels = 4 # Use only hue as input
    if args.use_gt_masks or args.use_gt_poses:
        assert(False)
    model = se3posenets.PoseMaskEncoder(
                    num_se3=args.num_se3, se3_type=args.se3_type,
                    input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                    init_se3_iden=args.init_posese3_iden,
                    use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                    sharpen_rate=args.sharpen_rate, wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                    num_state=args.num_state_net, noise_stop_iter=args.noise_stop_iter,
                    use_se3nn=args.use_se3nn)
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
            best_loss    = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
            best_floss   = checkpoint['best_flow_loss'] if 'best_flow_loss' in checkpoint else float("inf")
            best_fcloss  = checkpoint['best_flowconsis_loss'] if 'best_flowconsis_loss' in checkpoint else float("inf")
            best_epoch   = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
            best_fepoch  = checkpoint['best_flow_epoch'] if 'best_flow_epoch' in checkpoint else 0
            best_fcepoch = checkpoint['best_flowconsis_epoch'] if 'best_flowconsis_epoch' in checkpoint else 0
            print('==== Best validation loss: {} was from epoch: {} ===='.format(best_loss, best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_loss, best_floss, best_fcloss    = float("inf"), float("inf"), float("inf")
        best_epoch, best_fepoch, best_fcepoch = 0, 0, 0

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

    ## Create a file to log different validation errors over training epochs
    statstfile = open(args.save_dir + '/epochtrainstats.txt', 'w')
    statsvfile = open(args.save_dir + '/epochvalstats.txt', 'w')
    statstfile.write("Epoch, Loss, Ptloss, Consisloss, Flowerrsum, Flowerravg\n")
    statsvfile.write("Epoch, Loss, Ptloss, Consisloss, Flowerrsum, Flowerravg\n")

    ########################
    ############ Train / Validate
    args.imgdisp_freq = 5 * args.disp_freq # Tensorboard log frequency for the image data
    train_ids, val_ids = [], []
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs, args.min_lr)

        # Train for one epoch
        train_stats = iterate(train_loader, model, tblogger, args.train_ipe,
                           mode='train', optimizer=optimizer, epoch=epoch+1)
        train_ids += train_stats.data_ids

        # Evaluate on validation set
        val_stats = iterate(val_loader, model, tblogger, args.val_ipe,
                            mode='val', epoch=epoch+1)
        val_ids += val_stats.data_ids

        # Find best losses
        val_loss, val_floss, val_fcloss = val_stats.loss.avg, \
                                          val_stats.ptloss.avg.sum(), \
                                          val_stats.ptloss.avg.sum() + val_stats.consisloss.avg.sum()
        is_best, is_fbest, is_fcbest    = (val_loss < best_loss), (val_floss < best_floss), (val_fcloss < best_fcloss)
        prev_best_loss, prev_best_floss, prev_best_fcloss    = best_loss, best_floss, best_fcloss
        prev_best_epoch, prev_best_fepoch, prev_best_fcepoch = best_epoch, best_fepoch, best_fcepoch
        s, sf, sfc = 'SAME', 'SAME', 'SAME'
        if is_best:
            best_loss, best_epoch, s       = val_loss, epoch+1, 'IMPROVED'
        if is_fbest:
            best_floss, best_fepoch, sf    = val_floss, epoch+1, 'IMPROVED'
        if is_fcbest:
            best_fcloss, best_fcepoch, sfc = val_fcloss, epoch+1, 'IMPROVED'
        print('==== [LOSS]   Epoch: {}, Status: {}, Previous best: {:.5f}/{}. Current: {:.5f}/{} ===='.format(
                                    epoch+1, s, prev_best_loss, prev_best_epoch, best_loss, best_epoch))
        print('==== [FLOSS]  Epoch: {}, Status: {}, Previous best: {:.5f}/{}. Current: {:.5f}/{} ===='.format(
                                    epoch+1, sf, prev_best_floss, prev_best_fepoch, best_floss, best_fepoch))
        print('==== [FCLOSS] Epoch: {}, Status: {}, Previous best: {:.5f}/{}. Current: {:.5f}/{} ===='.format(
                                    epoch+1, sfc, prev_best_fcloss, prev_best_fcepoch, best_loss, best_fcepoch))

        # Write losses to stats file
        statstfile.write("{}, {}, {}, {}, {}, {}, {}\n".format(epoch+1, train_stats.loss.avg,
                                                                       train_stats.ptloss.avg.sum(),
                                                                       train_stats.consisloss.avg.sum(),
                                                                       train_stats.flowerr_sum.avg.sum()/args.batch_size,
                                                                       train_stats.flowerr_avg.avg.sum()/args.batch_size))
        statsvfile.write("{}, {}, {}, {}, {}, {}, {}\n".format(epoch + 1, val_stats.loss.avg,
                                                                       val_stats.ptloss.avg.sum(),
                                                                       val_stats.consisloss.avg.sum(),
                                                                       val_stats.flowerr_sum.avg.sum() / args.batch_size,
                                                                       val_stats.flowerr_avg.avg.sum() / args.batch_size))

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch+1,
            'args' : args,
            'best_loss'            : best_loss,
            'best_flow_loss'       : best_floss,
            'best_flowconsis_loss' : best_fcloss,
            'best_epoch'           : best_epoch,
            'best_flow_epoch'      : best_fepoch,
            'best_flowconsis_epoch': best_fcepoch,
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
        }, is_best, is_fbest, is_fcbest, savedir=args.save_dir, filename='checkpoint.pth.tar') #_{}.pth.tar'.format(epoch+1))
        print('\n')

    # Delete train and val data loaders
    del train_loader, val_loader

    # Load best model for testing (not latest one)
    print("=> loading best model from '{}'".format(args.save_dir + "/model_flow_best.pth.tar"))
    checkpoint = torch.load(args.save_dir + "/model_flow_best.pth.tar")
    num_train_iter = checkpoint['train_iter']
    try:
        model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded best checkpoint (epoch {}, train iter {})"
          .format(checkpoint['epoch'], num_train_iter))
    best_epoch   = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
    best_fepoch  = checkpoint['best_flow_epoch'] if 'best_flow_epoch' in checkpoint else 0
    best_fcepoch = checkpoint['best_flowconsis_epoch'] if 'best_flowconsis_epoch' in checkpoint else 0
    print('==== Best validation loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))
    print('==== Best validation flow loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_flow_loss'],
                                                                         best_fepoch))
    print('==== Best validation flow-consis loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_flowconsis_loss'],
                                                                         best_fcepoch))

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
    print('==== Best validation loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))
    print('==== Best validation flow loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_flow_loss'],
                                                                         best_fepoch))
    print('==== Best validation flow-consis loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_flowconsis_loss'],
                                                                         best_fcepoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'test_stats': {'stats': test_stats,
                       'niters': test_loader.niters, 'nruns': test_loader.nruns,
                       'totaliters': test_loader.iteration_count(),
                       'ids': test_stats.data_ids,
                       },
    }, is_best=False, savedir=args.save_dir, filename='test_stats.pth.tar')

    # Write test stats to val stats file at the end
    statsvfile.write("{}, {}, {}, {}, {}, {}, {}\n".format(checkpoint['epoch'], test_stats.loss.avg,
                                                                   test_stats.ptloss.avg.sum(),
                                                                   test_stats.consisloss.avg.sum(),
                                                                   test_stats.flowerr_sum.avg.sum() / args.batch_size,
                                                                   test_stats.flowerr_avg.avg.sum() / args.batch_size))
    statsvfile.close(); statstfile.close()

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
    stats.loss, stats.ptloss                    = AverageMeter(), AverageMeter()
    stats.flowerr_sum, stats.flowerr_avg        = AverageMeter(), AverageMeter()
    stats.motionerr_sum, stats.motionerr_avg    = AverageMeter(), AverageMeter()
    stats.stillerr_sum, stats.stillerr_avg      = AverageMeter(), AverageMeter()
    stats.maskconsisloss                        = AverageMeter()
    stats.data_ids = []
    if mode == 'test':
        # Save the flow errors and poses if in "testing" mode
        stats.motion_err, stats.motion_npt, stats.still_err, stats.still_npt = [], [], [], []

    # Switch model modes
    train = True if (mode == 'train') else False
    if train:
        assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
        model.train()
    else:
        assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}"+mode
        model.eval()

    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(
        mode, epoch, num_iters))
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    pt_wt = args.pt_wt * args.loss_scale
    mask_consis_wt = args.mask_consis_wt * args.loss_scale
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
        fwdflows = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        fwdvis   = util.to_var(sample['fwdvisibilities'].type(deftype), requires_grad=False)
        tarpts   = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        tarpts.data.add_(pts.data.narrow(1,0,1).expand_as(tarpts.data)) # Add "k"-step flows to the initial point cloud

        # Get XYZRGB input
        if args.use_xyzrgb:
            rgb = util.to_var(sample['rgbs'].type(deftype)/255.0, requires_grad=train) # Normalize RGB to 0-1
            netinput = torch.cat([pts, rgb], 2) # Concat along channels dimension
        elif args.use_xyzhue:
            hue = util.to_var(sample['rgbs'].narrow(2,0,1).type(deftype)/179.0, requires_grad=train)  # Normalize Hue to 0-1 (Opencv has hue from 0-179)
            netinput = torch.cat([pts, hue], 2) # Concat along channels dimension
        else:
            netinput = pts # XYZ

        # Get fwd pixel associations
        fwdpixelassocs = util.to_var(sample['fwdassocpixelids'].type(deftype), requires_grad=False)

        # Get jt angles
        jtangles = util.to_var(sample['actctrlconfigs'].type(deftype), requires_grad=train) #[:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        ####### Run a FWD pass through the network (multi-step)
        # Predict the poses and masks
        predposes, predmasks, predpremasks, preddeltas, predpts = [], [], [], [], []
        for k in xrange(pts.size(1)):
            # Predict the current pose and mask
            inp = [netinput[:,k], jtangles[:,k]] if args.use_jt_angles else netinput[:,k]
            currpose, currmask = model(inp, train_iter=num_train_iter, predict_masks=True)
            currpremask        = model.pre_sharpen_mask.clone()

            # Compute the deltas and predict pts forward in time
            if (k > 0):
                # Compute the deltas between t=0 and t=k - two ways: compose the deltas or compute difference to t0 each time
                if args.compose_deltas:
                    currdelta = se3.ComposeRtPair(currpose, se3.RtInverse(predposes[-1])) # p_k * p_(k-1)^-1
                    currdeltapose = currdelta if (k == 1) else se3.ComposeRtPair(currdelta, preddeltas[-1]) # p_k * p_(k-1)^-1 * p_(k-1) * p_0^-1
                else:
                    currdeltapose = se3.ComposeRtPair(currpose, se3.RtInverse(predposes[0])) # p_k * p_0^-1
                preddeltas.append(currdeltapose) # Save deltas

                # Predict the points forward in time
                # Takes pts at t = 0 and predicts them forward to time t = k
                nextpts = se3nn.NTfm3D()(pts[:,0], predmasks[0], currdeltapose)
                predpts.append(nextpts)

            # Save items
            predposes.append(currpose)
            predmasks.append(currmask)
            predpremasks.append(currpremask)

        ####### Compute losses - We use point loss in the FWD dirn and Consistency loss between poses
        ptloss, maskconsisloss = torch.zeros(args.seq_len), torch.zeros(args.seq_len)
        for k in xrange(args.seq_len):
            ### 3D loss
            # If motion-normalized loss, pass in GT flows
            inputs, targets = predpts[k] - pts[:,0], fwdflows[:,k]
            if args.motion_norm_loss:
                motion = targets  # Use either delta-flows or full-flows
                currptloss = pt_wt * ctrlnets.MotionNormalizedLoss3D(inputs, targets, motion=motion,
                                                                     loss_type=args.loss_type, wts=fwdvis[:,k])
            else:
                currptloss = pt_wt * ctrlnets.Loss3D(inputs, targets, loss_type=args.loss_type, wts=fwdvis[:,k])

            ### Mask Consistency Loss (between t & t+1)
            currmaskconsisloss = 0
            if mask_consis_wt > 0:
                if args.pre_mask_consis:
                    if args.mask_consis_loss_type == 'kl':
                        mask0in, mask1in = F.softmax(predpremasks[0]), F.softmax(predpremasks[k+1]) # Softmax the pre-sharpening activations
                    elif args.mask_consis_loss_type == 'kllog':
                        mask0in, mask1in = F.log_softmax(predpremasks[0]), F.log_softmax(predpremasks[k+1]) # Expects log inputs
                    else:
                        mask0in, mask1in = predpremasks[0], predpremasks[k+1]
                else:
                    assert(args.mask_consis_loss_type is not 'kllog') # We have mask outputs which are not logs
                    mask0in, mask1in = predmasks[0], predmasks[k+1]
                currmaskconsisloss = mask_consis_wt * mask_consistency_loss(mask0in, mask1in, fwdpixelassocs[:,k],
                                                                            args.mask_consis_loss_type)
                maskconsisloss[k] = currmaskconsisloss.data[0]

            # Append to total loss
            loss      = currptloss + currmaskconsisloss
            ptloss[k] = currptloss.data[0]

        # Update stats
        stats.ptloss.update(ptloss)
        stats.maskconsisloss.update(maskconsisloss)
        stats.loss.update(loss.data[0])

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

        # Display/Print frequency
        bsz = pts.size(0)
        if i % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                        samplecurr=j+1, sampletotal=len(data_loader),
                        stats=stats, bsz=bsz)

            ### Print stuff if we have weight sharpening enabled
            if args.use_wt_sharpening and not args.use_gt_masks:
                noise_std, pow = model.compute_wt_sharpening_stats(train_iter=num_train_iter)
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
                mode+'-flowerrsum': flowerr_sum.sum()/bsz,
                mode+'-flowerravg': flowerr_avg.sum()/bsz,
                mode+'-motionerrsum': motionerr_sum.sum()/bsz,
                mode+'-motionerravg': motionerr_avg.sum()/bsz,
                mode+'-stillerrsum': stillerr_sum.sum() / bsz,
                mode+'-stillerravg': stillerr_avg.sum() / bsz,
                mode+'-maskconsissloss': maskconsisloss.sum(),
            }
            if mode == 'train':
                info[mode+'-lr'] = args.curr_lr # Plot current learning rate
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
                    predpose  = predposes[k].data[id].cpu().float()
                    gtdepth   = normalize_img(sample['points'][id,k,2:].expand(3,args.img_ht,args.img_wd).permute(1,2,0), min=0, max=3)
                    for n in xrange(args.num_se3):
                        # Pose_1 (GT/Pred)
                        if n < gtpose.size(0):
                            util.draw_3d_frame(gtdepth, gtpose[n], [0,0,1], args.cam_intrinsics[0], pixlength=15.0) # GT pose: Blue
                        util.draw_3d_frame(gtdepth, predpose[n], [0,1,0], args.cam_intrinsics[0], pixlength=15.0) # Pred pose: Green
                    depths.append(gtdepth)
                depthdisp = torch.cat(depths, 1).permute(2,0,1) # Concatenate along columns (3 x 240 x 320*seq_len+1 image)

                # Concat the flows, depths and masks into one tensor
                flowdisp  = torchvision.utils.make_grid(torch.cat([flows.narrow(0,id,1),
                                                                   predflows.narrow(0,id,1)], 0).cpu().view(-1, 3, args.img_ht, args.img_wd),
                                                        nrow=args.seq_len, normalize=True, range=(-0.01, 0.01))
                #depthdisp = torchvision.utils.make_grid(sample['points'][id].narrow(1,2,1), normalize=True, range=(0.0,3.0))
                maskdisp  = torchvision.utils.make_grid(torch.cat([predmasks[0].data.narrow(0,id,1)], 0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                                                        nrow=args.num_se3, normalize=True, range=(0,1))

                # Display RGB
                if args.use_xyzrgb:
                    rgbdisp = torchvision.utils.make_grid(sample['rgbs'][id].float().view(-1, 3, args.img_ht, args.img_wd),
                                                            nrow=args.seq_len, normalize=True, range=(0.0,255.0))
                elif args.use_xyzhue:
                    rgbdisp = torchvision.utils.make_grid(sample['rgbs'][id,:,0].float().view(-1, 1, args.img_ht, args.img_wd),
                                                            nrow=args.seq_len, normalize=True, range=(0.0, 179.0)) # Show only hue, goes from 0-179 in OpenCV

                # Show as an image summary
                info = { mode+'-depths': util.to_np(depthdisp.unsqueeze(0)),
                         mode+'-flows' : util.to_np(flowdisp.unsqueeze(0)),
                         mode+'-masks' : util.to_np(maskdisp.narrow(0,0,1))
                }
                if args.use_xyzrgb or args.use_xyzhue:
                    info[mode+'-rgbs'] = util.to_np(rgbdisp.unsqueeze(0)) # Optional RGB
                for tag, images in info.items():
                    tblogger.image_summary(tag, images, iterct)

                ## Print the predicted delta-SE3s
                #deltase3s = deltapose01.data[id].view(args.num_se3, -1).cpu()
                #print('\tPredicted delta-SE3s from t=0-1:', deltase3s)

                ## Print the predicted mask values
                mask0 = predmasks[0]
                print('\tPredicted mask stats:')
                for k in xrange(args.num_se3):
                    print('\tMax: {:.4f}, Min: {:.4f}, Mean: {:.4f}, Std: {:.4f}, Median: {:.4f}, Pred 1: {}'.format(
                        mask0.data[id,k].max(), mask0.data[id,k].min(), mask0.data[id,k].mean(),
                        mask0.data[id,k].std(), mask0.data[id,k].view(-1).cpu().float().median(),
                        (mask0.data[id,k] - 1).abs().le(1e-5).sum()))
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
          'Loss: {loss.val:.4f} ({loss.avg:.4f}), '.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, bsz, loss=stats.loss))

    # Print flow loss per timestep
    for k in xrange(args.seq_len):
        print('\tStep: {}, Pt: {:.3f} ({:.3f}), '
              'Mask-Consis: {:.3f} ({:.4f}), '
              'Flow => Sum: {:.3f} ({:.3f}), Avg: {:.3f} ({:.3f}), '
              'Motion/Still => Sum: {:.3f}/{:.3f}, Avg: {:.3f}/{:.3f}'
            .format(
            1 + k * args.step_len,
            stats.ptloss.val[k], stats.ptloss.avg[k],
            stats.maskconsisloss.val[k], stats.maskconsisloss.avg[k],
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
    elif optim_type == 'asgd':
        optimizer = torch.optim.ASGD(params=parameters, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        assert False, "Unknown optimizer type: " + optim_type
    return optimizer

### Save checkpoint
def save_checkpoint(state, is_best, is_fbest=False, is_fcbest=False, savedir='.', filename='checkpoint.pth.tar'):
    savefile = savedir + '/' + filename
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, savedir + '/model_best.pth.tar')
    if is_fbest:
        shutil.copyfile(savefile, savedir + '/model_flow_best.pth.tar')
    if is_fcbest:
        shutil.copyfile(savefile, savedir + '/model_flowconsis_best.pth.tar')

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
def adjust_learning_rate(optimizer, epoch, decay_rate=0.1, decay_epochs=10, min_lr=1e-5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    lr = min_lr if (args.lr < min_lr) else lr # Clamp at min_lr
    print("======== Epoch: {}, Initial learning rate: {}, Current: {}, Min: {} =========".format(
        epoch, args.lr, lr, min_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.curr_lr = lr

### Gradient clipping hook
def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v

### Mask consistency error
# masks are B x K x H x W, assoc is B x 1 x H x W
def mask_consistency_loss(mask1, mask2, pixelassoc12, losstype='mse'):
    assert (losstype in ['mse', 'abs', 'kl', 'kllog'])
    bsz, nch, ht, wd = mask1.size()
    assert(mask1.is_same_size(mask2) and (pixelassoc12.size() == torch.Size([bsz, 1, ht, wd])))
    mask1v, mask2v, pixelassoc12v = mask1.contiguous().view(bsz, nch, ht*wd), \
                                    mask2.contiguous().view(bsz, nch, ht*wd), \
                                    pixelassoc12.contiguous().view(bsz, 1, ht*wd).long()
    assocmask = (pixelassoc12v != -1) # Only these points need to be penalized
    pixelassoc12v[pixelassoc12v == -1] = 0 # These points index the first value (doesn't matter as mask = 0 for those pts)
    mask2v_1 = torch.gather(mask2v, 2, pixelassoc12v.expand_as(mask1v)) # Vals from tensor 2 in tensor 1, associated correctly
    if losstype == 'mse':
        mseloss = (mask1v - mask2v_1).pow(2)
        maskloss = 0.5 * mseloss[assocmask].mean()
    elif losstype == 'abs':
        absloss = (mask1v - mask2v_1).abs()
        maskloss = absloss[assocmask].mean()
    elif losstype == 'kl':
        ## KL(inp, tar) = tar * (log(tar) - log(inp))
        nonzeros = util.to_var((mask1v.gt(0) * mask2v_1.gt(0) * assocmask).data.clone()) # Only compute loss for vals that are > 0, log is inf else
        klloss = mask2v_1 * (torch.log(mask2v_1) - torch.log(mask1v))
        maskloss = klloss[nonzeros].mean()
    elif losstype == 'kllog': # Assumes that you pass in log(inp) & log(tar)
        ## KL(inp, tar) = tar * (log(tar) - log(inp))
        klloss = torch.exp(mask2v_1) * (mask2v_1 - mask1v)
        maskloss = klloss[assocmask].mean()
    return maskloss

################ RUN MAIN
if __name__ == '__main__':
    main()
