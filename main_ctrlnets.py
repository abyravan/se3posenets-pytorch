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
import util
from util import AverageMeter, Tee, DataEnumerator

#### Setup options
# Common
import options
parser = options.setup_comon_options()

# Specific
parser.add_argument('--fwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the FWD direction (default: 1)')
parser.add_argument('--bwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the BWD direction (default: 1)')
parser.add_argument('--poswtavg-wt', default=0, type=float,
                    metavar='WT', help='Weight for the error between predicted position and mask weighted avg positions (default: 0)')
parser.add_argument('--seg-wt', default=0, type=float,
                    metavar='WT', help='Segmentation mask error (default: 0)')
parser.add_argument('--no-mask-gradmag', action='store_true', default=False,
                    help='uses only the loss gradient sign for training the masks (default: False)')

################ MAIN
#@profile
def main():
    # Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()
    args.batch_norm = not args.no_batch_norm
    assert (args.seq_len == 1), "Recurrent network training not enabled currently"

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

    # Setup extra options
    args.img_ht, args.img_wd, args.img_suffix = 240, 320, 'sub'
    args.num_ctrl = 14 if (args.ctrl_type.find('both') >= 0) else 7 # Number of control dimensions
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data
    load_dir = args.data[0] #args.data.split(',,')[0]
    args.baxter_labels = data.read_baxter_labels_file(load_dir + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']
    args.cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
    args.se3_dim = ctrlnets.get_se3_dimension(args.se3_type, args.pred_pivot)
    print('Predicting {} SE3s of type: {}. Dim: {}'.format(args.num_se3, args.se3_type, args.se3_dim))

    # Camera parameters
    args.cam_intrinsics = {'fx': 589.3664541825391/2,
                           'fy': 589.3664541825391/2,
                           'cx': 320.5/2,
                           'cy': 240.5/2}
    args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               args.cam_intrinsics)

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => FWD: {}, BWD: {}, CONSIS: {}'.format(
        args.loss_scale, args.fwd_wt, args.bwd_wt, args.consis_wt))
    if args.poswtavg_wt > 0:
        print('Loss weight for position error w.r.t mask wt. avg positions: {}'.format(args.poswtavg_wt))

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

    # Loss type
    if args.use_wt_sharpening or args.use_sigmoid_mask:
        norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
        print('3D loss type: ' + args.loss_type + norm_motion)

    # NTFM3D-Delta
    if args.use_ntfm_delta:
        print('Using the variant of NTFM3D that computes a weighted avg. flow per point using the SE3 transforms')

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    # Seg loss
    if args.seg_wt > 0:
        assert args.num_se3==2, "Segmentation loss only works for 2 SE3s currently"
        print('Adding a segmentation loss based on flow magnitude. Loss weight: {}'.format(args.seg_wt))

    # Mask gradient magnitude
    args.use_mask_gradmag = not args.no_mask_gradmag
    if args.no_mask_gradmag:
        assert (not args.use_sigmoid_mask or args.use_ntfm_delta), "Option to not use mask gradient magnitudes is not possible with sigmoid-masking/NTFM delta"
        print("Using only the gradient's sign for training the masks. Discarding the magnitude")

    # TODO: Add option for using encoder pose for tfm t2

    ########################
    ############ Load datasets
    # Get datasets
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                       img_scale = args.img_scale, ctrl_type = 'actdiffvel',
                                                                       mesh_ids = args.mesh_ids,
                                                                       camera_extrinsics = args.cam_extrinsics,
                                                                       camera_intrinsics = args.cam_intrinsics,
                                                                       compute_bwdflows=True)
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')   # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
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

    ### Load the model
    num_train_iter = 0
    model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                  se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                  input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                  init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                                  use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                                  sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                                  use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                                  wide=args.wide_model)
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    ### Load optimizer
    optimizer = load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        # TODO: Save path to TB log dir, save new log there again
        # TODO: Reuse options in args (see what all to use and what not)
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
        iterate(test_loader, model, tblogger, len(test_loader), mode='test')
        return # Finish

    ########################
    ############ Train / Validate
    best_val_loss, best_epoch = float("inf"), 0
    args.imgdisp_freq = 5 * args.disp_freq # Tensorboard log frequency for the image data
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs)

        # Train for one epoch
        tr_loss, tr_fwdloss, tr_bwdloss, tr_consisloss, tr_poswtavgloss, \
            tr_flowsum_f, tr_flowavg_f, \
            tr_flowsum_b, tr_flowavg_b = iterate(train_loader, model, tblogger, args.train_ipe,
                                                 mode='train', optimizer=optimizer, epoch=epoch+1)

        # Log values and gradients of the parameters (histogram)
        # NOTE: Doing this in the loop makes the stats file super large / tensorboard processing slow
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            tblogger.histo_summary(tag, util.to_np(value.data), epoch + 1)
            tblogger.histo_summary(tag + '/grad', util.to_np(value.grad), epoch + 1)

        # Evaluate on validation set
        val_loss, val_fwdloss, val_bwdloss, val_consisloss, val_poswtavgloss, \
            val_flowsum_f, val_flowavg_f, \
            val_flowsum_b, val_flowavg_b = iterate(val_loader, model, tblogger, args.val_ipe,
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
            'train_stats': {'loss': tr_loss, 'fwdloss': tr_fwdloss, 'bwdloss': tr_bwdloss,
                            'consisloss': tr_consisloss, 'poswtavgloss': tr_poswtavgloss,
                            'flowsum_f': tr_flowsum_f, 'flowavg_f': tr_flowavg_f,
                            'flowsum_b': tr_flowsum_b, 'flowavg_b': tr_flowavg_b,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count()
                            },
            'val_stats'  : {'loss': val_loss, 'fwdloss': val_fwdloss, 'bwdloss': val_bwdloss,
                            'consisloss': val_consisloss, 'poswtavgloss': val_poswtavgloss,
                            'flowsum_f': val_flowsum_f, 'flowavg_f': val_flowavg_f,
                            'flowsum_b': val_flowsum_b, 'flowavg_b': val_flowavg_b,
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
    iterate(test_loader, model, tblogger, len(test_loader), mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Close log file
    #logfile.close()

################# HELPER FUNCTIONS

### Main iterate function (train/test/val)
def iterate(data_loader, model, tblogger, num_iters,
            mode='test', optimizer=None, epoch=0):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    lossm, ptlossm_f, ptlossm_b, consislossm = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    seglossm_f, seglossm_b = AverageMeter(), AverageMeter()
    poswtavglossm  = AverageMeter()
    flowlossm_sum_f, flowlossm_avg_f = AverageMeter(), AverageMeter()
    flowlossm_sum_b, flowlossm_avg_b = AverageMeter(), AverageMeter()

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

    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(
        mode, epoch, num_iters))
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    for i in xrange(num_iters):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        j, sample = data_loader.next()

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts_1    = util.to_var(sample['points'][:, 0].clone().type(deftype), requires_grad=True)
        pts_2    = util.to_var(sample['points'][:, 1].clone().type(deftype), requires_grad=True)
        ctrls_1  = util.to_var(sample['controls'][:, 0].clone().type(deftype), requires_grad=True)
        fwdflows = util.to_var(sample['fwdflows'][:, 0].clone().type(deftype), requires_grad=False)
        bwdflows = util.to_var(sample['bwdflows'][:, 0].clone().type(deftype), requires_grad=False)
        tarpts_1 = util.to_var((sample['points'][:, 0] + sample['fwdflows'][:, 0]).type(deftype), requires_grad=False)
        tarpts_2 = util.to_var((sample['points'][:, 1] + sample['bwdflows'][:, 0]).type(deftype), requires_grad=False)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        # Run the FWD pass through the network
        [pose_1, mask_1], [pose_2, mask_2], [deltapose_t_12, pose_t_2] = model([pts_1, pts_2, ctrls_1],
                                                                               train_iter=num_train_iter)
        deltapose_t_21 = se3nn.RtInverse()(deltapose_t_12)  # Invert the delta-pose

        # Compute predicted points based on the masks
        # TODO: Add option for using encoder pose for tfm t2
        predpts_1 = ptpredlayer(use_mask_gradmag=args.use_mask_gradmag)(pts_1, mask_1, deltapose_t_12)
        predpts_2 = ptpredlayer(use_mask_gradmag=args.use_mask_gradmag)(pts_2, mask_2, deltapose_t_21)

        # Compute 3D point loss (3D losses @ t & t+1)
        # For soft mask model, compute losses without predicting points. Otherwise use predicted pts
        fwd_wt, bwd_wt, consis_wt = args.fwd_wt * args.loss_scale, args.bwd_wt * args.loss_scale, args.consis_wt * args.loss_scale
        if args.use_wt_sharpening or args.use_sigmoid_mask:
            # Choose inputs & target for losses. Normalized MSE losses operate on flows, others can work on 3D points
            inputs_1, targets_1 = [predpts_1-pts_1, tarpts_1-pts_1] if (args.loss_type == 'normmsesqrt') else [predpts_1, tarpts_1]
            inputs_2, targets_2 = [predpts_2-pts_2, tarpts_2-pts_2] if (args.loss_type == 'normmsesqrt') else [predpts_2, tarpts_2]
            # We just measure error in flow space
            #inputs_1, targets_1 = (predpts_1 - pts_1), fwdflows
            #inputs_2, targets_2 = (predpts_2 - pts_2), bwdflows
            # If motion-normalized loss, pass in GT flows
            if args.motion_norm_loss:
                ptloss_1 = fwd_wt * ctrlnets.MotionNormalizedLoss3D(inputs_1, targets_1, fwdflows, loss_type=args.loss_type)
                ptloss_2 = bwd_wt * ctrlnets.MotionNormalizedLoss3D(inputs_2, targets_2, bwdflows, loss_type=args.loss_type)
            else:
                ptloss_1 = fwd_wt * ctrlnets.Loss3D(inputs_1, targets_1, loss_type=args.loss_type)
                ptloss_2 = bwd_wt * ctrlnets.Loss3D(inputs_2, targets_2, loss_type=args.loss_type)
        else:
            # Use the weighted 3D transform loss, do not use explicitly predicted points
            ptloss_1 = fwd_wt * se3nn.Weighted3DTransformLoss(use_mask_gradmag=args.use_mask_gradmag)(pts_1, mask_1, deltapose_t_12, tarpts_1)  # Predict pts in FWD dirn and compare to target @ t2
            ptloss_2 = bwd_wt * se3nn.Weighted3DTransformLoss(use_mask_gradmag=args.use_mask_gradmag)(pts_2, mask_2, deltapose_t_21, tarpts_2)  # Predict pts in BWD dirn and compare to target @ t1

        # Compute pose consistency loss
        consisloss = consis_wt * ctrlnets.BiMSELoss(pose_2, pose_t_2)  # Enforce consistency between pose @ t1 predicted by encoder & pose @ t1 from transition model

        # Compute loss between the predicted positions and the weighted avg of the masks & point clouds
        poswtavgloss = 0
        if args.poswtavg_wt > 0:
            # Compute the weighted average position @ t & t+1 based on the predicted masks & the point clouds
            wtavgpos_1 = se3nn.WeightedAveragePoints()(pts_1, mask_1)
            wtavgpos_2 = se3nn.WeightedAveragePoints()(pts_2, mask_2)

            # Compute the loss as a sum of the position error @ t & t+1
            # Compute error between the predicted position & the weighted average position based on the masks & pt clouds
            # Pred poses are: B x nSE3 x 3 x 4, Wt avg poses are: B x nSE3 x 3 x 1
            poswtavgloss_1 = ctrlnets.BiMSELoss(pose_1.narrow(3, 3, 1), wtavgpos_1)
            poswtavgloss_2 = ctrlnets.BiMSELoss(pose_2.narrow(3, 3, 1), wtavgpos_2)
            poswtavgloss = 0.5 * args.poswtavg_wt * args.loss_scale * (poswtavgloss_1 + poswtavgloss_2) # Sum up losses
            poswtavglossm.update(poswtavgloss.data[0]) # Update avg. meter

        # Use a segmentation loss to encourage better segmentation
        # Pts that have flow > 0 are bunched into a single mask & rest in another mask
        segloss_1, segloss_2 = 0, 0
        if args.seg_wt > 0:
            seg_wt = args.seg_wt * args.loss_scale
            masktargs_1, masktargs_2 = fwdflows.abs().sum(1).gt(0).squeeze().long(), bwdflows.abs().sum(1).gt(0).squeeze().long()
            segloss_1 = seg_wt * nn.CrossEntropyLoss(size_average=True)(mask_1, masktargs_1)
            segloss_2 = seg_wt * nn.CrossEntropyLoss(size_average=True)(mask_2, masktargs_2)
            seglossm_f.update(segloss_1.data[0]); seglossm_b.update(segloss_2.data[0])

        # Compute total loss as sum of all losses
        loss = ptloss_1 + ptloss_2 + consisloss + poswtavgloss + segloss_1 + segloss_2

        # Update stats
        ptlossm_f.update(ptloss_1.data[0]); ptlossm_b.update(ptloss_2.data[0])
        consislossm.update(consisloss.data[0]); lossm.update(loss.data[0])

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
        predfwdflows = (predpts_1 - pts_1)
        predbwdflows = (predpts_2 - pts_2)
        flowloss_sum_fwd, flowloss_avg_fwd, _, _ = compute_flow_errors(predfwdflows.data.unsqueeze(1),
                                                                           fwdflows.data.unsqueeze(1))
        flowloss_sum_bwd, flowloss_avg_bwd, _, _ = compute_flow_errors(predbwdflows.data.unsqueeze(1),
                                                                           bwdflows.data.unsqueeze(1))

        # Update stats
        flowlossm_sum_f.update(flowloss_sum_fwd); flowlossm_sum_b.update(flowloss_sum_bwd)
        flowlossm_avg_f.update(flowloss_avg_fwd); flowlossm_avg_b.update(flowloss_avg_bwd)
        #numptsm_f.update(numpts_fwd); numptsm_b.update(numpts_bwd)
        #nzm_f.update(nz_fwd); nzm_b.update(nz_bwd)

        # Display/Print frequency
        if i % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                        samplecurr=j+1, sampletotal=len(data_loader),
                        loss=lossm, fwdloss=ptlossm_f, bwdloss=ptlossm_b,
                        consisloss=consislossm, poswtavgloss=poswtavglossm,
                        flowloss_sum_f=flowlossm_sum_f, flowloss_sum_b=flowlossm_sum_b,
                        flowloss_avg_f=flowlossm_avg_f, flowloss_avg_b=flowlossm_avg_b)
            print('\tSegLoss-Fwd: {fwd.val:.5f} ({fwd.avg:.5f}), '
                  'SegLoss-Bwd: {bwd.val:.5f} ({bwd.avg:.5f})'.format(
                fwd=seglossm_f, bwd=seglossm_b))

            ### Print stuff if we have weight sharpening enabled
            if args.use_wt_sharpening:
                noise_std, pow = model.posemaskmodel.compute_wt_sharpening_stats(train_iter=num_train_iter)
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
                mode+'-fwd3dloss': ptloss_1.data[0],
                mode+'-bwd3dloss': ptloss_2.data[0],
                mode+'-consisloss': consisloss.data[0],
                mode+'-poswtavgloss': poswtavglossm.val
            }
            for tag, value in info.items():
                tblogger.scalar_summary(tag, value, iterct)

            # (2) Log images & print predicted SE3s
            # TODO: Numpy or matplotlib
            if i % args.imgdisp_freq == 0:

                ## Log the images (at a lower rate for now)
                id = random.randint(0, sample['points'].size(0)-1)

                # Concat the flows, depths and masks into one tensor
                flowdisp  = torchvision.utils.make_grid(torch.cat([fwdflows.data.narrow(0,id,1),
                                                                   bwdflows.data.narrow(0,id,1),
                                                                   predfwdflows.data.narrow(0,id,1),
                                                                   predbwdflows.data.narrow(0,id,1)], 0).cpu(),
                                                        nrow=2, normalize=True, range=(-0.01, 0.01))
                depthdisp = torchvision.utils.make_grid(sample['points'][id].narrow(1,2,1), normalize=True, range=(0.0,3.0))
                maskdisp  = torchvision.utils.make_grid(torch.cat([mask_1.data.narrow(0,id,1),
                                                                   mask_2.data.narrow(0,id,1)], 0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                                                        nrow=args.num_se3, normalize=True, range=(0,1))
                # Show as an image summary
                info = { mode+'-depths': util.to_np(depthdisp.narrow(0,0,1)),
                         mode+'-flows' : util.to_np(flowdisp.unsqueeze(0)),
                         mode+'-masks' : util.to_np(maskdisp.narrow(0,0,1))
                }
                for tag, images in info.items():
                    tblogger.image_summary(tag, images, iterct)

                ## Print the predicted delta-SE3s
                print '\tPredicted delta-SE3s @ t=2:', predictions['deltase3'].data[id].view(args.num_se3,
                                                                                             args.se3_dim).cpu()

                ## Print the predicted mask values
                print('\tPredicted mask stats:')
                for k in xrange(args.num_se3):
                    print('\tMax: {:.4f}, Min: {:.4f}, Mean: {:.4f}, Std: {:.4f}, Median: {:.4f}, Pred 1: {}'.format(
                        mask_1.data[id,k].max(), mask_1.data[id,k].min(), mask_1.data[id,k].mean(),
                        mask_1.data[id,k].std(), mask_1.data[id,k].view(-1).cpu().float().median(),
                        (mask_1.data[id,k] - 1).abs().le(1e-5).sum()))
                print('')

        # Measure viz time
        viz_time.update(time.time() - start)

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=num_iters, total=num_iters,
                samplecurr=data_loader.niters+1, sampletotal=len(data_loader),
                loss=lossm, fwdloss=ptlossm_f, bwdloss=ptlossm_b,
                consisloss=consislossm, poswtavgloss=poswtavglossm,
                flowloss_sum_f=flowlossm_sum_f, flowloss_sum_b=flowlossm_sum_b,
                flowloss_avg_f=flowlossm_avg_f, flowloss_avg_b=flowlossm_avg_b)
    print('\tSegLoss-Fwd: {fwd.val:.5f} ({fwd.avg:.5f}), '
          'SegLoss-Bwd: {bwd.val:.5f} ({bwd.avg:.5f})'.format(
        fwd=seglossm_f, bwd=seglossm_b))
    print('========================================================')

    # Return the loss & flow loss
    return lossm, ptlossm_f, ptlossm_b, consislossm, poswtavglossm, \
           flowlossm_sum_f, flowlossm_avg_f, \
           flowlossm_sum_b, flowlossm_avg_b

### Print statistics
def print_stats(mode, epoch, curr, total, samplecurr, sampletotal,
                loss, fwdloss, bwdloss, consisloss, poswtavgloss,
                flowloss_sum_f, flowloss_avg_f,
                flowloss_sum_b, flowloss_avg_b):
    # Print loss
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], '
          'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
          'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
          'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
          'Consis: {consis.val:.3f} ({consis.avg:.3f}), '
          'PosWtAvg: {poswtavg.val:.3f} ({poswtavg.avg:.3f})'.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, loss=loss, fwd=fwdloss, bwd=bwdloss,
        consis=consisloss, poswtavg=poswtavgloss))

    # Print flow loss per timestep
    bsz = args.batch_size
    for k in xrange(args.seq_len):
        print('\tStep: {}, Fwd => Sum: {:.3f} ({:.3f}), Avg: {:.6f} ({:.6f}), '
              'Bwd => Sum: {:.3f} ({:.3f}), Avg: {:.6f} ({:.6f})'.format(
            1 + k * args.step_len,
            flowloss_sum_f.val[k] / bsz, flowloss_sum_f.avg[k] / bsz,
            flowloss_avg_f.val[k] / bsz, flowloss_avg_f.avg[k] / bsz,
            flowloss_sum_b.val[k] / bsz, flowloss_sum_b.avg[k] / bsz,
            flowloss_avg_b.val[k] / bsz, flowloss_avg_b.avg[k] / bsz))

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
    num_pts, loss_sum, loss_avg, nz = torch.zeros(seq), torch.zeros(seq), torch.zeros(seq), torch.zeros(seq)
    for k in xrange(seq):
        # Compute errors per dataset
        # !!!!!!!!! :ge(1e-3) returns a ByteTensor and if u sum within byte tensors, the max value we can get is 255 !!!!!!!!!
        num_pts_d  = torch.abs(gtflows[:,k]).sum(1).ge(1e-3).float().view(batch, -1).sum(1) # Num pts with flow per dataset
        loss_sum_d = (predflows[:,k] - gtflows[:,k]).pow(2).view(batch, -1).sum(1).float()  # Sum of errors per dataset

        # Sum up errors per batch
        num_pts[k]  = num_pts_d.sum()               # Num pts that have non-zero flow
        loss_sum[k] = loss_sum_d.sum()              # Sum of total flow loss across the batch
        for j in xrange(batch):
            if (num_pts_d[j] > 0):
                loss_avg[k] += (loss_sum_d[j] / num_pts_d[j]) # Sum of per-point loss across the batch
                nz[k]       += 1 # We have one more dataset with non-zero num pts that move
    # Return
    return loss_sum, loss_avg, num_pts, nz

### Compute flow errors per mask (flows are size: B x S x 3 x H x W)
def compute_flow_errors_per_mask(predflows, gtflows, gtmasks):
    batch, seq, nse3 = predflows.size(0), predflows.size(1), gtmasks.size(2)  # B x S x 3 x H x W
    num_pts, loss_sum, loss_avg, nz = torch.zeros(seq,nse3), torch.zeros(seq,nse3), torch.zeros(seq,nse3), torch.zeros(seq,nse3)
    for k in xrange(seq):
        mask = torch.abs(gtflows[:,k]).sum(1).ge(1e-3).float()        # Set of points that move in the current scene
        err  = (predflows[:,k] - gtflows[:,k]).pow(2).sum(1).float()  # Flow error for current step
        for j in xrange(nse3):  # Iterate over the mask-channels
            # Compute error per dataset
            maskc       = gtmasks[:,j].clone().float() * mask   # Pts belonging to current link that move in scene
            num_pts_d   = gtmasks[:,j].clone().view(batch, -1).sum(1).float() # Num pts per mask per dataset
            loss_sum_d  = (err * maskc).view(batch, -1).sum(1)  # Flow error sum per mask per dataset

            # Sum up errors actoss the batch
            num_pts[k][j]   = num_pts_d.sum()   # Num pts that have non-zero flow
            loss_sum[k][j]  = loss_sum_d.sum()  # Sum of total flow loss across batch
            for i in xrange(batch):
                if (num_pts_d[i] > 0):
                    loss_avg[k][j]  += (loss_sum_d[i] / num_pts_d[i]) # Sum of per-point flow across batch
                    nz[k][j]        += 1 # One more dataset
    # Return
    return loss_sum, loss_avg, num_pts, nz

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
