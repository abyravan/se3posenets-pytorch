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

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import se2nets
import util
from util import AverageMeter, Tee, DataEnumerator

#### Setup options
# Common
import options
parser = options.setup_comon_options()

# Loss options
parser.add_argument('--delta-flow-loss', action='store_true', default=False,
                    help='Penalize only the current flow per unroll of the network rather than penalizing the entire flow for each unroll (default: False)')
parser.add_argument('--pt-wt', default=1, type=float,
                    metavar='WT', help='Weight for the 3D point loss - only FWD direction (default: 1)')

# K-step consistency
parser.add_argument('--kstep-consis', action='store_true', default=False,
                    help='Consistency between t = 0 & t = k+1 instead of t = k & t = k+1 (default: False)')


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

    # Get default options & camera intrinsics
    load_dir = args.data[0] #args.data.split(',,')[0]
    try:
        # Read from file
        intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
        print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
        if args.se2_data:
            args.img_ht, args.img_wd = int(intrinsics['ht']), int(intrinsics['wd'])
        else:
            args.img_ht, args.img_wd = 240, 320  # All data except SE(2) data is at 240x320 resolution
        args.img_scale = 1.0 / intrinsics['s']  # Scale of the image (use directly from the data)

        # Setup camera intrinsics
        sc = float(args.img_ht) / intrinsics['ht']  # Scale factor for the intrinsics
        args.cam_intrinsics = {'fx': intrinsics['fx'] * sc,
                               'fy': intrinsics['fy'] * sc,
                               'cx': intrinsics['cx'] * sc,
                               'cy': intrinsics['cy'] * sc}
        print("Scale factor for the intrinsics: {}".format(sc))
    except:
        print("Could not read intrinsics file, reverting to default settings")
        args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
        args.cam_intrinsics = {'fx': 589.3664541825391 / 2,
                               'fy': 589.3664541825391 / 2,
                               'cx': 320.5 / 2,
                               'cy': 240.5 / 2}
    print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(args.img_ht, args.img_wd,
                                                                                args.cam_intrinsics['fx'],
                                                                                args.cam_intrinsics['fy'],
                                                                                args.cam_intrinsics['cx'],
                                                                                args.cam_intrinsics['cy']))

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

    # Compute intrinsic grid
    args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               args.cam_intrinsics)
    # Image suffix
    args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data
    args.baxter_labels = data.read_statelabels_file(load_dir + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']
    args.cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
    args.se3_dim = ctrlnets.get_se3_dimension(args.se3_type, args.pred_pivot)
    print('Predicting {} SE3s of type: {}. Dim: {}'.format(args.num_se3, args.se3_type, args.se3_dim))

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
        args.loss_scale, args.pt_wt, args.consis_wt))
    print('Dissimilarity loss weights => POSE: {}, DELTA: {}'.format(
        args.pose_dissim_wt, args.delta_dissim_wt))

    # Soft-weight sharpening
    if args.soft_wt_sharpening:
        print('Using weight sharpening with soft-masking loss')
        assert not args.use_sigmoid_mask, "Cannot set both soft weight sharpening and sigmoid mask options together"
        args.use_wt_sharpening = True # We do weight sharpening!

    # Weight sharpening stuff
    if args.use_wt_sharpening:
        print('Using weight sharpening to encourage binary mask prediction. Start iter: {}, Rate: {}'.format(
            args.sharpen_start_iter, args.sharpen_rate))
        assert not args.use_sigmoid_mask, "Cannot set both weight sharpening and sigmoid mask options together"
    elif args.use_sigmoid_mask:
        print('Using sigmoid to generate masks, treating each channel independently. A pixel can belong to multiple masks now')
    else:
        print('Using soft-max + weighted 3D transform loss to encourage mask prediction')
        assert not args.motion_norm_loss, "Cannot use normalized-motion losses along with soft-masking"
        if (args.loss_type.find('normmsesqrt') < 0):
            assert not args.delta_flow_loss,  "Can only use delta-flow losses + soft-masking with Normalized MSE loss"

    # Loss type
    delta_loss = ', Penalizing the delta-flow loss per unroll' if args.delta_flow_loss else ''
    norm_motion = ''
    if args.use_wt_sharpening or args.use_sigmoid_mask:
        norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
    print('3D loss type: ' + args.loss_type + norm_motion + delta_loss)

    # NTFM3D-Delta
    if args.use_ntfm_delta:
        print('Using the variant of NTFM3D that computes a weighted avg. flow per point using the SE3 transforms')

    # Decomp model
    if args.decomp_model:
        assert args.seq_len > 1, "Decomposed pose/mask encoders can be used only with multi-step models"

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    if args.use_jt_angles:
        print("Using Jt angles as input to the pose encoder")

    if args.use_jt_angles_trans:
        print("Using Jt angles as input to the transition model")

    if args.kstep_consis:
        print("Using k-step consistency loss")

    # TODO: Add option for using encoder pose for tfm t2

    # DA threshold / winsize
    print("Flow/visibility computation. DA threshold: {}, DA winsize: {}".format(args.da_thresh,
                                                                                 args.da_winsize))

    ########################
    ############ Load datasets
    # Get datasets
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                       img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                                       num_ctrl=args.num_ctrl, num_state=args.num_state,
                                                                       mesh_ids = args.mesh_ids, ctrl_ids=ctrlids_in_state,
                                                                       camera_extrinsics = args.cam_extrinsics,
                                                                       camera_intrinsics = args.cam_intrinsics,
                                                                       compute_bwdflows=args.use_gt_masks, num_tracker=args.num_tracker,
                                                                       dathreshold=args.da_thresh, dawinsize=args.da_winsize) # Need BWD flows / masks if using GT masks
    filter_func = lambda b: data.filter_func(b, mean_dt=args.mean_dt, std_dt=args.std_dt)
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train', filter_func)  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val',   filter_func)  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test',  filter_func)  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # Create a data-collater for combining the samples of the data into batches along with some post-processing
    # TODO: Batch along dim 1 instead of dim 0

    # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
    train_loader = DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                        collate_fn=train_dataset.collate_batch))
    val_loader   = DataEnumerator(util.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                        collate_fn=val_dataset.collate_batch))

    ########################
    ############ Load models & optimization stuff

    if args.se2_data:
        print('Using the smaller multi-step SE2-Pose-Model')
    else:
        print('Using multi-step SE3-Pose-Model')

    ### Load the model
    num_train_iter = 0
    if args.use_gt_masks:
        print('Using GT masks. Model predicts only poses & delta-poses')
        assert not args.use_gt_poses, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.MultiStepSE2OnlyPoseModel if args.se2_data else ctrlnets.MultiStepSE3OnlyPoseModel
    elif args.use_gt_poses:
        assert NotImplementedError # We don't have stuff for this
        print('Using GT poses & delta poses. Model predicts only masks')
        assert not args.use_gt_masks, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.MultiStepSE2OnlyMaskModel if args.se2_data else ctrlnets.MultiStepSE3OnlyMaskModel
    else:
        modelfn = se2nets.MultiStepSE2PoseModel if args.se2_data else ctrlnets.MultiStepSE3PoseModel
    model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                    se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                    input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                    init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                    use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                    sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv, decomp_model=args.decomp_model,
                    use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                    wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                    use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state)
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
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ########################
    ############ Test (don't create the data loader unless needed, creates 4 extra threads)
    if args.evaluate:
        # Delete train and val loaders
        del train_loader, val_loader

        # TODO: Move this to before the train/val loader creation??
        print('==== Evaluating pre-trained network on test data ===')
        args.imgdisp_freq = 10 * args.disp_freq  # Tensorboard log frequency for the image data
        sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
        test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, sampler=sampler, pin_memory=args.use_pin_memory,
                                        collate_fn=test_dataset.collate_batch))
        te_loss, te_ptloss, te_consisloss, \
            te_flowsum, te_flowavg = iterate(test_loader, model, tblogger, len(test_loader), mode='test')

        # Save final test error
        save_checkpoint({
            'args': args,
            'test_stats': {'loss': te_loss, 'ptloss': te_ptloss, 'consisloss': te_consisloss,
                           'flowsum': te_flowsum, 'flowavg': te_flowavg,
                           'niters': test_loader.niters, 'nruns': test_loader.nruns,
                           'totaliters': test_loader.iteration_count()
                           },
        }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

        # Close log file & return
        logfile.close()
        return

    ########################
    ############ Train / Validate
    best_val_loss, best_epoch = float("inf"), 0
    args.imgdisp_freq = 5 * args.disp_freq # Tensorboard log frequency for the image data
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs)

        # Train for one epoch
        tr_loss, tr_ptloss, tr_consisloss, \
            tr_flowsum, tr_flowavg = iterate(train_loader, model, tblogger, args.train_ipe,
                                             mode='train', optimizer=optimizer, epoch=epoch+1)

        # Log values and gradients of the parameters (histogram)
        # NOTE: Doing this in the loop makes the stats file super large / tensorboard processing slow
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            tblogger.histo_summary(tag, util.to_np(value.data), epoch + 1)
            tblogger.histo_summary(tag + '/grad', util.to_np(value.grad), epoch + 1)

        # Evaluate on validation set
        val_loss, val_ptloss, val_consisloss, \
            val_flowsum, val_flowavg = iterate(val_loader, model, tblogger, args.val_ipe,
                                                   mode='val', epoch=epoch+1)

        # Find best loss
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
            'train_stats': {'loss': tr_loss, 'ptloss': tr_ptloss,
                            'consisloss': tr_consisloss,
                            'flowsum': tr_flowsum, 'flowavg': tr_flowavg,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count()
                            },
            'val_stats'  : {'loss': val_loss, 'ptloss': val_ptloss,
                            'consisloss': val_consisloss,
                            'flowsum': val_flowsum, 'flowavg': val_flowavg,
                            'niters': val_loader.niters, 'nruns': val_loader.nruns,
                            'totaliters': val_loader.iteration_count()
                            },
            'train_iter' : num_train_iter,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, is_best, savedir=args.save_dir)
        print('\n')

    # Delete train and val data loaders
    del train_loader, val_loader

    # Do final testing (if not asked to evaluate)
    # (don't create the data loader unless needed, creates 4 extra threads)
    print('==== Evaluating trained network on test data ====')
    args.imgdisp_freq = 10 * args.disp_freq # Tensorboard log frequency for the image data
    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, sampler=sampler, pin_memory=args.use_pin_memory,
                                    collate_fn=test_dataset.collate_batch))
    te_loss, te_ptloss, te_consisloss, \
        te_flowsum, te_flowavg = iterate(test_loader, model, tblogger, len(test_loader),
                                         mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'test_stats': {'loss': te_loss, 'ptloss': te_ptloss, 'consisloss': te_consisloss,
                       'flowsum': te_flowsum, 'flowavg': te_flowavg,
                       'niters': test_loader.niters, 'nruns': test_loader.nruns,
                       'totaliters': test_loader.iteration_count()
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
    lossm, ptlossm, consislossm = AverageMeter(), AverageMeter(), AverageMeter()
    flowlossm_sum, flowlossm_avg = AverageMeter(), AverageMeter()
    dissimposelossm, dissimdeltalossm = AverageMeter(), AverageMeter()
    #flowlossm_mask_sum, flowlossm_mask_avg = AverageMeter(), AverageMeter()
    consiserrorm = AverageMeter()

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
    ptpredlayer = se3nn.NTfm3DDelta if args.use_ntfm_delta else se3nn.NTfm3D

    # Type of loss (mixture of experts = wt sharpening or sigmoid)
    mex_loss = (args.use_wt_sharpening or args.use_sigmoid_mask) and (not args.soft_wt_sharpening)

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

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts      = util.to_var(sample['points'].type(deftype), requires_grad=train)
        ctrls    = util.to_var(sample['controls'].type(deftype), requires_grad=train)
        jtangles = util.to_var(sample['actconfigs'].type(deftype), requires_grad=train)
        fwdflows = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        fwdvis   = util.to_var(sample['fwdvisibilities'].type(deftype), requires_grad=False)
        tarpts   = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        tarpts.data.add_(pts.data.narrow(1,0,1).expand_as(tarpts.data)) # Add "k"-step flows to the initial point cloud

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        ### Run a FWD pass through the network (multi-step)
        # Predict the poses and masks
        poses, initmask = [], None
        for k in xrange(pts.size(1)):
            # Predict the pose and mask at time t = 0
            # For all subsequent timesteps, predict only the poses
            if(k == 0):
                if args.use_gt_masks:
                    p = model.forward_only_pose([pts[:,k], jtangles[:,k]]) # We can only predict poses
                    initmask = util.to_var(sample['masks'][:,0].type(deftype).clone(), requires_grad=False) # Use GT masks
                else:
                    p, initmask = model.forward_pose_mask([pts[:,k], jtangles[:,k]], train_iter=num_train_iter)
            else:
                p = model.forward_only_pose([pts[:,k], jtangles[:,k]])
            poses.append(p)

        # Make next-pose predictions & corresponding 3D point predictions using the transition model
        # We use pose_0 and [ctrl_0, ctrl_1, .., ctrl_(T-1)] to make the predictions
        deltaposes, transposes = [], []
        compdeltaposes = []
        for k in xrange(args.seq_len):
            # Get current pose
            if (k == 0):
                pose = poses[0] # Use initial pose
            else:
                pose = transposes[k-1] # Use previous predicted pose

            # Predict next pose based on curr pose, control
            delta, trans = model.forward_next_pose(pose, ctrls[:,k], jtangles[:,k])
            deltaposes.append(delta)
            transposes.append(trans)

            # Compose the deltas over time (T4 = T4 * T3^-1 * T3 * T2^-1 * T2 * T1^-1 * T1 = delta_4 * delta_3 * delta_2 * T1
            if (k == 0):
                compdeltaposes.append(delta)
            else:
                compdeltaposes.append(se3nn.ComposeRtPair()(delta, compdeltaposes[k-1])) # delta_full = delta_curr * delta_prev

        # Now compute the losses across the sequence
        # We use point loss in the FWD dirn and Consistency loss between poses
        # For the point loss, we use the initial point cloud and mask &
        # predict in a sequence based on the predicted changes in poses
        predpts, ptloss, consisloss, loss = [], torch.zeros(args.seq_len), torch.zeros(args.seq_len), 0
        dissimposeloss, dissimdeltaloss = torch.zeros(args.seq_len), torch.zeros(args.seq_len)
        for k in xrange(args.seq_len):
            # Get current point cloud
            if (k == 0):
                currpts = pts[:,0]  # Use initial point cloud
            else:
                currpts = predpts[k-1]  # Use previous predicted point cloud

            # Predict transformed point cloud based on the previous point cloud & new delta-transform
            if args.delta_flow_loss:
                nextpts = ptpredlayer()(currpts, initmask, deltaposes[k])
            else:
                nextpts = ptpredlayer()(pts[:,0], initmask, compdeltaposes[k])
            predpts.append(nextpts)

            # Compute 3D point loss
            # For soft mask model, compute losses without predicting points (using composed transforms). Otherwise use predicted pts
            if not mex_loss:
                # For weighted 3D transform loss, it is enough to set the mask values of not-visible pixels to all zeros
                # These pixels will end up having zero loss then
                vismask = initmask * fwdvis[:, k] # Set all not-visible pixels to 0 mask, = 0 loss

                # Use the weighted 3D transform loss, do not use explicitly predicted points
                if (args.loss_type.find('normmsesqrt') >= 0):
                    if args.delta_flow_loss:
                        # Transform from pt_k-1 -> pt_k & check against error w.r.t current step's flows
                        tgtflows = fwdflows[:,k] - (0 if (k == 0) else fwdflows[:,k-1]) # Flow for those points in that step alone!
                        currptloss = pt_wt * se3nn.Weighted3DTransformNormLoss()(currpts, vismask,
                                                                                 deltaposes[k], tgtflows)
                    else:
                        # Transform from pt_0 -> pt_k & check against error w.r.t full flows
                        currptloss = pt_wt * se3nn.Weighted3DTransformNormLoss()(pts[:,0], vismask,
                                                                                 compdeltaposes[k], fwdflows[:,k])
                else:
                    currptloss = pt_wt * se3nn.Weighted3DTransformLoss()(currpts, vismask, deltaposes[k], tarpts[:, k])  # Weighted point loss!
            else:
                # Choose inputs & target for losses. Normalized MSE losses operate on flows, others can work on 3D points
                if args.delta_flow_loss:
                    # For each step, we only look at the target flow for that step (how much do those points move based on that control alone)
                    # and compare that against the predicted flows for that step alone!
                    inputs  = nextpts - currpts # Delta flow for that step
                    targets = fwdflows[:,k] - (0 if (k == 0) else fwdflows[:,k-1]) # Flow for those points in that step alone!
                else:
                    inputs, targets = [nextpts-pts[:,0], tarpts[:,k]-pts[:,0]] if (args.loss_type == 'normmsesqrt') else [nextpts, tarpts[:,k]]
                # If motion-normalized loss, pass in GT flows
                if args.motion_norm_loss:
                    motion  = targets if args.delta_flow_loss else fwdflows[:,k] # Use either delta-flows or full-flows
                    currptloss = pt_wt * ctrlnets.MotionNormalizedLoss3D(inputs, targets, motion=motion,
                                                                         loss_type=args.loss_type, wts=fwdvis[:, k])
                else:
                    currptloss = pt_wt * ctrlnets.Loss3D(inputs, targets, loss_type=args.loss_type, wts=fwdvis[:, k])

            # Compute pose consistency loss
            if args.kstep_consis:
                # Consistency between t = 0 & t = k+1 always
                if args.no_consis_delta_grads:
                    delta = util.to_var(compdeltaposes[k].data.clone(), requires_grad=False)  # Break the graph here
                else:
                    delta = compdeltaposes[k]  # Don't break the graph
                nextpose_trans = se3nn.ComposeRtPair()(delta, poses[0])  # pose_k = pose_k * pose_0^-1 * pose_0
                currconsisloss = consis_wt * ctrlnets.BiMSELoss(nextpose_trans, poses[k+1])  # Enforce consistency between pose predicted by encoder & pose from transition model
            else:
                # Consistency between t & t+1 always
                if args.no_consis_delta_grads:
                    delta = util.to_var(deltaposes[k].data.clone(), requires_grad=False)  # Break the graph here
                    nextpose_trans = se3nn.ComposeRtPair()(delta, poses[k])
                    currconsisloss = consis_wt * ctrlnets.BiMSELoss(nextpose_trans, poses[k+1])  # Enforce consistency between pose predicted by encoder & pose from transition model
                else:
                    currconsisloss = consis_wt * ctrlnets.BiMSELoss(transposes[k], poses[k+1])  # Enforce consistency between pose predicted by encoder & pose from transition model

            # Add a loss for pose dis-similarity & delta dis-similarity
            dissimpose_wt, dissimdelta_wt = args.pose_dissim_wt * args.loss_scale, args.delta_dissim_wt * args.loss_scale
            currdissimposeloss  = dissimpose_wt * ctrlnets.DisSimilarityLoss(poses[k][:,1:],
                                                                             poses[k+1][:,1:],
                                                                             size_average=True)  # Enforce dis-similarity in pose space
            currdissimdeltaloss = dissimdelta_wt * ctrlnets.DisSimilarityLoss(deltaposes[k][:,1:],
                                                                      identfm.expand_as(deltaposes[k][:,1:]),
                                                                      size_average=True) # Change in pose > 0

            # Append to total loss
            loss += currptloss + currconsisloss + currdissimposeloss + currdissimdeltaloss
            ptloss[k]     = currptloss.data[0]
            consisloss[k] = currconsisloss.data[0]
            dissimposeloss[k]  = currdissimposeloss.data[0]
            dissimdeltaloss[k] = currdissimdeltaloss.data[0]

        # Update stats
        ptlossm.update(ptloss)
        consislossm.update(consisloss)
        lossm.update(loss.data[0])
        dissimposelossm.update(dissimposeloss)
        dissimdeltalossm.update(dissimdeltaloss)

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
        flowloss_sum, flowloss_avg, _, _ = compute_flow_errors(predflows, flows)

        # Update stats
        flowlossm_sum.update(flowloss_sum); flowlossm_avg.update(flowloss_avg)

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
        consiserrorm.update(consiserror)

        # Display/Print frequency
        if i % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                        samplecurr=j+1, sampletotal=len(data_loader),
                        loss=lossm, ptloss=ptlossm, consisloss=consislossm,
                        posedisloss=dissimposelossm, deltadisloss=dissimdeltalossm,
                        flowloss_sum=flowlossm_sum, flowloss_avg=flowlossm_avg,
                        consiserror=consiserrorm, bsz=pts.size(0))

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
            print('\tNum datasets sampled/discarded: {}/{}'.format(data_loader.data.dataset.numsampled,
                                                                 data_loader.data.dataset.numdiscarded))
            ### TensorBoard logging
            # (1) Log the scalar values
            iterct = data_loader.iteration_count() # Get total number of iterations so far
            bsz = pts.size(0)
            info = {
                mode+'-loss': loss.data[0],
                mode+'-pt3dloss': ptloss.sum(),
                mode+'-consisloss': consisloss.sum(),
                mode+'-dissimposeloss': dissimposeloss.sum(),
                mode+'-dissimdeltaloss': dissimdeltaloss.sum(),
                mode+'-consiserror': consiserror.sum(),
                mode+'-consiserrormax': consiserrormax.sum(),
                mode+'-flowlosssum': flowloss_sum.sum()/bsz,
                mode+'-flowlossavg': flowloss_avg.sum()/bsz,
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
                            util.draw_3d_frame(gtdepth, gtpose[n],     [0,0,1], args.cam_intrinsics, pixlength=15.0) # GT pose: Blue
                        util.draw_3d_frame(gtdepth, predpose[n],   [0,1,0], args.cam_intrinsics, pixlength=15.0) # Pred pose: Green
                        if predposet is not None:
                            util.draw_3d_frame(gtdepth, predposet[n], [1,0,0], args.cam_intrinsics, pixlength=15.0)  # Transition model pred pose: Red
                    depths.append(gtdepth)
                depthdisp = torch.cat(depths, 1).permute(2,0,1) # Concatenate along columns (3 x 240 x 320*seq_len+1 image)

                # Concat the flows, depths and masks into one tensor
                flowdisp  = torchvision.utils.make_grid(torch.cat([flows.narrow(0,id,1),
                                                                   predflows.narrow(0,id,1)], 0).cpu().view(-1, 3, args.img_ht, args.img_wd),
                                                        nrow=args.seq_len, normalize=True, range=(-0.01, 0.01))
                #depthdisp = torchvision.utils.make_grid(sample['points'][id].narrow(1,2,1), normalize=True, range=(0.0,3.0))
                maskdisp  = torchvision.utils.make_grid(torch.cat([initmask.data.narrow(0,id,1)], 0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                                                        nrow=args.num_se3, normalize=True, range=(0,1))
                # Show as an image summary
                info = { mode+'-depths': util.to_np(depthdisp.unsqueeze(0)),
                         mode+'-flows' : util.to_np(flowdisp.unsqueeze(0)),
                         mode+'-masks' : util.to_np(maskdisp.narrow(0,0,1))
                }
                for tag, images in info.items():
                    tblogger.image_summary(tag, images, iterct)

                ## Print the predicted delta-SE3s
                if not args.use_gt_poses:
                    print '\tPredicted delta-SE3s @ t=2:', predictions['deltase3'].data[id].view(args.num_se3,
                                                                                                 args.se3_dim).cpu()

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
                loss=lossm, ptloss=ptlossm, consisloss=consislossm,
                posedisloss=dissimposelossm, deltadisloss=dissimdeltalossm,
                flowloss_sum=flowlossm_sum, flowloss_avg=flowlossm_avg,
                consiserror=consiserrorm)
    print('========================================================')

    # Return the loss & flow loss
    return lossm, ptlossm, consislossm, \
           flowlossm_sum, flowlossm_avg

### Print statistics
def print_stats(mode, epoch, curr, total, samplecurr, sampletotal,
                loss, ptloss, consisloss, posedisloss, deltadisloss,
                flowloss_sum, flowloss_avg, consiserror, bsz=None):
    # Print loss
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], '
          'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, loss=loss))

    # Print flow loss per timestep
    bsz = args.batch_size if bsz is None else bsz
    for k in xrange(args.seq_len):
        print('\tStep: {}, Pt: {:.3f} ({:.3f}), Consis: {:.3f}/{:.4f} ({:.3f}/{:.4f}), '
              'Pose-Dissim: {:.3f} ({:.3f}), Delta-Dissim: {:.3f} ({:.3f}), '
              'Flow => Sum: {:.3f} ({:.3f}), Avg: {:.6f} ({:.6f}), '.format(
            1 + k * args.step_len,
            ptloss.val[k], ptloss.avg[k], consisloss.val[k], consisloss.avg[k],
            consiserror.val[k], consiserror.avg[k],
            posedisloss.val[k], posedisloss.avg[k], deltadisloss.val[k], deltadisloss.avg[k],
            flowloss_sum.val[k] / bsz, flowloss_sum.avg[k] / bsz,
            flowloss_avg.val[k] / bsz, flowloss_avg.avg[k] / bsz))

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

### Compute flow errors (flows are size: B x S x 3 x H x W)
def compute_flow_errors(predflows, gtflows):
    batch, seq = predflows.size(0), predflows.size(1) # B x S x 3 x H x W
    # Compute num pts not moving per mask
    # !!!!!!!!! > 1e-3 returns a ByteTensor and if u sum within byte tensors, the max value we can get is 255 !!!!!!!!!
    num_pts_d = (gtflows.abs().sum(2) > 1e-3).type_as(gtflows).view(batch,seq,-1).sum(2) # B x seq
    num_pts, nz = num_pts_d.sum(0), (num_pts_d > 0).type_as(gtflows).sum(0)
    # Compute loss across batch
    loss_sum_d = (predflows - gtflows).pow(2).view(batch,seq,-1).sum(2)  # Flow error for current step (B x seq)
    # Compute avg loss per example in the batch
    loss_avg_d = loss_sum_d / num_pts_d
    loss_avg_d[loss_avg_d != loss_avg_d] = 0  # Clear out any Nans
    loss_avg_d[loss_avg_d == np.inf] = 0  # Clear out any Infs
    loss_sum, loss_avg = loss_sum_d.sum(0), loss_avg_d.sum(0)
    # Return
    return loss_sum.cpu().float(), loss_avg.cpu().float(), num_pts.cpu().float(), nz.cpu().float()

### Compute flow errors per mask (flows are size: B x S x 3 x H x W)
def compute_flow_errors_per_mask(predflows, gtflows, gtmasks):
    batch, seq, nse3 = predflows.size(0), predflows.size(1), gtmasks.size(2)  # B x S x 3 x H x W
    # Compute num pts not moving per mask
    num_pts_d = gtmasks.type_as(gtflows).view(batch, seq, nse3, -1).sum(3)
    num_pts, nz = num_pts_d.sum(0), (num_pts_d > 0).type_as(gtflows).sum(0)
    # Compute loss across batch
    err  = (predflows - gtflows).pow(2).sum(2).unsqueeze(2).expand_as(gtmasks) # Flow error for current step (B x S x K x H x W)
    loss_sum_d = (err * gtmasks).view(batch, seq, nse3, -1).sum(3) # Flow error sum for all masks in entire sequence per dataset
    # Compute avg loss per example in the batch
    loss_avg_d = loss_sum_d / num_pts_d
    loss_avg_d[loss_avg_d != loss_avg_d] = 0 # Clear out any Nans
    loss_avg_d[loss_avg_d == np.inf]     = 0 # Clear out any Infs
    loss_sum, loss_avg = loss_sum_d.sum(0), loss_avg_d.sum(0)
    # Return
    return loss_sum.cpu().float(), loss_avg.cpu().float(), num_pts.cpu().float(), nz.cpu().float()

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
