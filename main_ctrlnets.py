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

# Specific
parser.add_argument('--fwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the FWD direction (default: 1)')
parser.add_argument('--bwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the BWD direction (default: 1)')

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

    # Get default options & camera intrinsics
    load_dir = args.data[0] #args.data.split(',,')[0]
    try:
        intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
        print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
        args.img_ht, args.img_wd, args.img_scale = int(intrinsics['ht']), int(intrinsics['wd']), 1.0/intrinsics['s']
        args.cam_intrinsics = {'fx': intrinsics['fx'],
                               'fy': intrinsics['fy'],
                               'cx': intrinsics['cx'],
                               'cy': intrinsics['cy']}
    except:
        print("Could not read intrinsics file, reverting to default settings")
        args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
        args.cam_intrinsics = {'fx': 589.3664541825391 / 2,
                               'fy': 589.3664541825391 / 2,
                               'cx': 320.5 / 2,
                               'cy': 240.5 / 2}

    # Compute intrinsic grid
    args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               args.cam_intrinsics)
    # Image suffix
    args.num_state = args.num_ctrl # Assume this
    args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data
    args.baxter_labels = data.read_baxter_labels_file(load_dir + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']
    args.cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
    args.se3_dim = ctrlnets.get_se3_dimension(args.se3_type, args.pred_pivot)
    print('Predicting {} SE3s of type: {}. Dim: {}'.format(args.num_se3, args.se3_type, args.se3_dim))

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => FWD: {}, BWD: {}, CONSIS: {}'.format(
        args.loss_scale, args.fwd_wt, args.bwd_wt, args.consis_wt))
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

    # Loss type
    if args.use_wt_sharpening or args.use_sigmoid_mask:
        norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
        print('3D loss type: ' + args.loss_type + norm_motion)
    else:
        assert not (args.loss_type == 'abs'), "No abs loss available for soft-masking"
        print('3D loss type: ' + args.loss_type)

    # Normalize per pt
    args.norm_per_pt = (args.loss_type == 'normmsesqrtpt')
    if args.norm_per_pt:
        print('Normalizing MSE error per 3D point instead of per dimension')

    # NTFM3D-Delta
    mex_loss = (args.use_wt_sharpening or args.use_sigmoid_mask) and (not args.soft_wt_sharpening)
    if args.use_ntfm_delta:
        assert mex_loss, "Cannot do NTFM3D + Delta for the Soft-Losses"
        print('Using the variant of NTFM3D that computes a weighted avg. flow per point using the SE3 transforms')

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    # Get IDs of joints we move (based on the data folders)
    args.jt_ids = []
    if args.se2_data:
        args.jt_ids = [x for x in xrange(args.num_se3-1)]
    elif args.num_se3 == 8:
        args.jt_ids = [0,1,2,3,4,5,6] # All joints
    else:
        for dir in args.data:
            if dir.find('joint') >= 0:
                id = int(dir[dir.find('joint')+5]) # Find the ID of the joint
                args.jt_ids.append(id)
    assert(args.num_se3 == len(args.jt_ids)+1) # This has to match (only if using single joint data!)
    args.jt_ids = np.sort(args.jt_ids) # Sort in ascending order
    print('Using data where the following joints move: ', args.jt_ids)

    # TODO: Add option for using encoder pose for tfm t2

    ########################
    ############ Load datasets
    # Get datasets
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                       img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                                       num_ctrl=args.num_ctrl, num_state=args.num_state,
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

    if args.se2_data:
        print('Using the smaller SE2 models')

    ### Load the model
    num_train_iter = 0
    if args.use_gt_masks:
        print('Using GT masks. Model predicts only poses & delta-poses')
        assert not args.use_gt_poses, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.SE2OnlyPoseModel if args.se2_data else ctrlnets.SE3OnlyPoseModel
    elif args.use_gt_poses:
        print('Using GT poses & delta poses. Model predicts only masks')
        assert not args.use_gt_masks, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.SE2OnlyMaskModel if args.se2_data else ctrlnets.SE3OnlyMaskModel
    else:
        if args.decomp_model:
            print('Decomp model. Training separate networks for pose & mask prediction')
            modelfn = se2nets.SE2DecompModel if args.se2_data else ctrlnets.SE3DecompModel
        else:
            print('Networks for pose & mask prediction share conv parameters')
            modelfn = se2nets.SE2PoseModel if args.se2_data else ctrlnets.SE3PoseModel
    model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
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
        te_loss, te_fwdloss, te_bwdloss, te_consisloss, \
            te_flowsum_f, te_flowavg_f, \
            te_flowsum_b, te_flowavg_b = iterate(test_loader, model, tblogger, len(test_loader), mode='test')

        # Save final test error
        save_checkpoint({
            'args': args,
            'test_stats': {'loss': te_loss, 'fwdloss': te_fwdloss, 'bwdloss': te_bwdloss, 'consisloss': te_consisloss,
                           'flowsum_f': te_flowsum_f, 'flowavg_f': te_flowavg_f,
                           'flowsum_b': te_flowsum_b, 'flowavg_b': te_flowavg_b,
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
        tr_loss, tr_fwdloss, tr_bwdloss, tr_consisloss, \
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
        val_loss, val_fwdloss, val_bwdloss, val_consisloss, \
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
            'train_stats': {'loss': tr_loss, 'fwdloss': tr_fwdloss, 'bwdloss': tr_bwdloss, 'consisloss': tr_consisloss,
                            'flowsum_f': tr_flowsum_f, 'flowavg_f': tr_flowavg_f,
                            'flowsum_b': tr_flowsum_b, 'flowavg_b': tr_flowavg_b,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count()
                            },
            'val_stats'  : {'loss': val_loss, 'fwdloss': val_fwdloss, 'bwdloss': val_bwdloss, 'consisloss': val_consisloss,
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
    te_loss, te_fwdloss, te_bwdloss, te_consisloss, \
        te_flowsum_f, te_flowavg_f, \
        te_flowsum_b, te_flowavg_b = iterate(test_loader, model, tblogger, len(test_loader),
                                             mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'best_loss': best_val_loss,
        'test_stats': {'loss': te_loss, 'fwdloss': te_fwdloss, 'bwdloss': te_bwdloss, 'consisloss': te_consisloss,
                        'flowsum_f': te_flowsum_f, 'flowavg_f': te_flowavg_f,
                        'flowsum_b': te_flowsum_b, 'flowavg_b': te_flowavg_b,
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
    lossm, ptlossm_f, ptlossm_b, consislossm = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    dissimposelossm, dissimdeltalossm = AverageMeter(), AverageMeter()
    flowlossm_sum_f, flowlossm_avg_f = AverageMeter(), AverageMeter()
    flowlossm_sum_b, flowlossm_avg_b = AverageMeter(), AverageMeter()
    flowlossm_mask_sum_f, flowlossm_mask_avg_f = AverageMeter(), AverageMeter()
    flowlossm_mask_sum_b, flowlossm_mask_avg_b = AverageMeter(), AverageMeter()
    deltaroterrm, deltatranserrm = AverageMeter(), AverageMeter()
    deltaroterrm_mask, deltatranserrm_mask = AverageMeter(), AverageMeter()

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
    if not args.use_gt_poses:
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
    identfm = util.to_var(torch.eye(4).view(1,1,4,4).expand(1,args.num_se3-1,4,4).narrow(2,0,3).type(deftype), requires_grad=False)
    for i in xrange(num_iters):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        j, sample = data_loader.next()

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts_1    = util.to_var(sample['points'][:, 0].clone().type(deftype), requires_grad=train)
        pts_2    = util.to_var(sample['points'][:, 1].clone().type(deftype), requires_grad=train)
        ctrls_1  = util.to_var(sample['controls'][:, 0].clone().type(deftype), requires_grad=train)
        fwdflows = util.to_var(sample['fwdflows'][:, 0].clone().type(deftype), requires_grad=False)
        bwdflows = util.to_var(sample['bwdflows'][:, 0].clone().type(deftype), requires_grad=False)
        fwdvis   = util.to_var(sample['fwdvisibilities'][:, 0].clone().type(deftype), requires_grad=False)
        bwdvis   = util.to_var(sample['bwdvisibilities'][:, 0].clone().type(deftype), requires_grad=False)
        tarpts_1 = util.to_var((sample['points'][:, 0] + sample['fwdflows'][:, 0]).type(deftype), requires_grad=False)
        tarpts_2 = util.to_var((sample['points'][:, 1] + sample['bwdflows'][:, 0]).type(deftype), requires_grad=False)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        # Run the FWD pass through the network
        if args.use_gt_masks:
            masks = get_jt_masks(sample['masks'].type(deftype), args.jt_ids)
            mask_1 = util.to_var(masks[:,0].clone(), requires_grad=False)
            mask_2 = util.to_var(masks[:,1].clone(), requires_grad=False)
            pose_1, pose_2, [deltapose_t_12, pose_t_2] = model([pts_1, pts_2, ctrls_1])
        elif args.use_gt_poses:
            pose_1 = util.to_var(get_jt_poses(sample['poses'][:, 0], args.jt_ids).clone().type(deftype), requires_grad=False) # 0 is BG, so we add 1
            pose_2 = util.to_var(get_jt_poses(sample['poses'][:, 1], args.jt_ids).clone().type(deftype), requires_grad=False) # 0 is BG, so we add 1
            deltapose_t_12, pose_t_2 = se3nn.ComposeRtPair()(pose_2, se3nn.RtInverse()(pose_1)), pose_2  # Pose_t+1 * Pose_t^-1
            mask_1, mask_2 = model([pts_1, pts_2], train_iter=num_train_iter)
        else:
            [pose_1, mask_1], [pose_2, mask_2], [deltapose_t_12, pose_t_2] = model([pts_1, pts_2, ctrls_1],
                                                                                   train_iter=num_train_iter)
        deltapose_t_21 = se3nn.RtInverse()(deltapose_t_12)  # Invert the delta-pose

        # Compute predicted points based on the masks
        # TODO: Add option for using encoder pose for tfm t2
        predpts_1 = ptpredlayer()(pts_1, mask_1, deltapose_t_12)
        predpts_2 = ptpredlayer()(pts_2, mask_2, deltapose_t_21)

        # Compute 3D point loss (3D losses @ t & t+1)
        # For soft mask model, compute losses without predicting points. Otherwise use predicted pts
        fwd_wt, bwd_wt, consis_wt = args.fwd_wt * args.loss_scale, args.bwd_wt * args.loss_scale, args.consis_wt * args.loss_scale
        if not mex_loss:
            # For weighted 3D transform loss, it is enough to set the mask values of not-visible pixels to all zeros
            # These pixels will end up having zero loss then
            vismask_1 = mask_1 * fwdvis # Should be broadcasted properly
            vismask_2 = mask_2 * bwdvis # Should be broadcasted properly

            # Use the weighted 3D transform loss, do not use explicitly predicted points
            if (args.loss_type.find('normmsesqrt') >= 0):
                ptloss_1 = fwd_wt * se3nn.Weighted3DTransformNormLoss(norm_per_pt=args.norm_per_pt)(pts_1, vismask_1, deltapose_t_12, fwdflows)  # Predict pts in FWD dirn and compare to target @ t2
                ptloss_2 = bwd_wt * se3nn.Weighted3DTransformNormLoss(norm_per_pt=args.norm_per_pt)(pts_2, vismask_2, deltapose_t_21, bwdflows)  # Predict pts in BWD dirn and compare to target @ t1
            else:
                ptloss_1 = fwd_wt * se3nn.Weighted3DTransformLoss()(pts_1, vismask_1, deltapose_t_12, tarpts_1)  # Predict pts in FWD dirn and compare to target @ t2
                ptloss_2 = bwd_wt * se3nn.Weighted3DTransformLoss()(pts_2, vismask_2, deltapose_t_21, tarpts_2)  # Predict pts in BWD dirn and compare to target @ t1
        else:
            # Choose inputs & target for losses. Normalized MSE losses operate on flows, others can work on 3D points
            inputs_1, targets_1 = [predpts_1-pts_1, tarpts_1-pts_1] if (args.loss_type.find('normmsesqrt') >= 0) else [predpts_1, tarpts_1]
            inputs_2, targets_2 = [predpts_2-pts_2, tarpts_2-pts_2] if (args.loss_type.find('normmsesqrt') >= 0) else [predpts_2, tarpts_2]
            # If motion-normalized loss, pass in GT flows
            # We weight each pixel's loss by it's visibility, so not-visible pixels will get zero loss
            if args.motion_norm_loss:
                ptloss_1 = fwd_wt * ctrlnets.MotionNormalizedLoss3D(inputs_1, targets_1, fwdflows,
                                                                    loss_type=args.loss_type, wts=fwdvis)
                ptloss_2 = bwd_wt * ctrlnets.MotionNormalizedLoss3D(inputs_2, targets_2, bwdflows,
                                                                    loss_type=args.loss_type, wts=bwdvis)
            else:
                ptloss_1 = fwd_wt * ctrlnets.Loss3D(inputs_1, targets_1, loss_type=args.loss_type, wts=fwdvis)
                ptloss_2 = bwd_wt * ctrlnets.Loss3D(inputs_2, targets_2, loss_type=args.loss_type, wts=bwdvis)

        # Compute pose consistency loss
        consisloss = consis_wt * ctrlnets.BiMSELoss(pose_2, pose_t_2)  # Enforce consistency between pose @ t1 predicted by encoder & pose @ t1 from transition model

        # Add a loss for pose dis-similarity & delta dis-similarity
        dissimpose_wt, dissimdelta_wt = args.pose_dissim_wt * args.loss_scale, args.delta_dissim_wt * args.loss_scale
        dissimposeloss  = dissimpose_wt  * ctrlnets.DisSimilarityLoss(pose_1[:,1:],
                                                                      pose_2[:,1:],
                                                                      size_average=True) # Enforce dis-similarity in pose space
        dissimdeltaloss = dissimdelta_wt * ctrlnets.DisSimilarityLoss(deltapose_t_12[:,1:],
                                                                      identfm.expand_as(deltapose_t_12[:,1:]),
                                                                      size_average=True) # Change in pose > 0

        # Compute total loss as sum of all losses
        loss = ptloss_1 + ptloss_2 + consisloss + dissimposeloss + dissimdeltaloss

        # Update stats
        ptlossm_f.update(ptloss_1.data[0]); ptlossm_b.update(ptloss_2.data[0])
        consislossm.update(consisloss.data[0]); lossm.update(loss.data[0])
        dissimposelossm.update(dissimposeloss.data[0]); dissimdeltalossm.update(dissimdeltaloss.data[0])

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

        ### Flow error
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

        # Compute flow error per mask (if asked to)
        if args.disp_err_per_mask:
            if args.use_gt_masks: # If we have all joints
                gtmask_1, gtmask_2 = mask_1.data, mask_2.data # Already computed
            else:
                gtmask_1 = get_jt_masks(sample['masks'][:, 0].type(deftype), args.jt_ids)
                gtmask_2 = get_jt_masks(sample['masks'][:, 1].type(deftype), args.jt_ids)
            flowloss_mask_sum_fwd, flowloss_mask_avg_fwd, _, _ = compute_flow_errors_per_mask(predfwdflows.data.unsqueeze(1),
                                                                                              fwdflows.data.unsqueeze(1),
                                                                                              gtmask_1.unsqueeze(1))
            flowloss_mask_sum_bwd, flowloss_mask_avg_bwd, _, _ = compute_flow_errors_per_mask(predbwdflows.data.unsqueeze(1),
                                                                                              bwdflows.data.unsqueeze(1),
                                                                                              gtmask_2.unsqueeze(1))

            # Update stats
            flowlossm_mask_sum_f.update(flowloss_mask_sum_fwd); flowlossm_mask_sum_b.update(flowloss_mask_sum_bwd)
            flowlossm_mask_avg_f.update(flowloss_mask_avg_fwd); flowlossm_mask_avg_b.update(flowloss_mask_avg_bwd)

        ### Pose error
        # Compute error in delta pose space (not used for backprop)
        gtpose_1 = get_jt_poses(sample['poses'][:, 0], args.jt_ids)
        gtpose_2 = get_jt_poses(sample['poses'][:, 1], args.jt_ids)
        gtdeltapose_t_12 = data.ComposeRtPair(gtpose_2, data.RtInverse(gtpose_1))  # Pose_t+1 * Pose_t^-1
        deltaroterr, deltatranserr = compute_pose_errors(deltapose_t_12.data.unsqueeze(1).cpu(),
                                                         gtdeltapose_t_12.unsqueeze(1))
        deltaroterrm.update(deltaroterr[0]); deltatranserrm.update(deltatranserr[0])

        # Compute rot & trans err per pose channel
        if args.disp_err_per_mask:
            deltaroterr_mask, deltatranserr_mask = compute_pose_errors_per_mask(deltapose_t_12.data.unsqueeze(1).cpu(),
                                                                                gtdeltapose_t_12.unsqueeze(1))
            deltaroterrm_mask.update(deltaroterr_mask);  deltatranserrm_mask.update(deltatranserr_mask)

        ### Display/Print frequency
        if i % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                        samplecurr=j+1, sampletotal=len(data_loader),
                        loss=lossm, fwdloss=ptlossm_f, bwdloss=ptlossm_b, consisloss=consislossm,
                        flowloss_sum_f=flowlossm_sum_f, flowloss_sum_b=flowlossm_sum_b,
                        flowloss_avg_f=flowlossm_avg_f, flowloss_avg_b=flowlossm_avg_b)
            print('\tPose-Dissim loss: {disP.val:.5f} ({disP.avg:.5f}), '
                  'Delta-Dissim loss: {disD.val:.5f} ({disD.avg:.5f}), '
                  'Delta-Rot-Err: {delR.val:.5f} ({delR.avg:.5f}), '
                  'Delta-Trans-Err: {delt.val:.5f} ({delt.avg:.5f})'.format(
                  disP=dissimposelossm, disD=dissimdeltalossm, delR=deltaroterrm, delt=deltatranserrm))

            # Print (flow & pose) error per mask if enabled
            if args.disp_err_per_mask:
                bsz = args.batch_size
                for k in xrange(args.num_se3):
                    print('\tSE3/Mask: {}, Fwd => Sum: {:.3f} ({:.3f}), Avg: {:.6f} ({:.6f}), '
                          'Bwd => Sum: {:.3f} ({:.3f}), Avg: {:.6f} ({:.6f}), '
                          'Delta-Rot: {:.5f} ({:.5f}), Delta-Trans: {:.5f} ({:.5f})'.format(
                            k,
                            flowlossm_mask_sum_f.val[0,k] / bsz, flowlossm_mask_sum_f.avg[0,k] / bsz,
                            flowlossm_mask_avg_f.val[0,k] / bsz, flowlossm_mask_avg_f.avg[0,k] / bsz,
                            flowlossm_mask_sum_b.val[0,k] / bsz, flowlossm_mask_sum_b.avg[0,k] / bsz,
                            flowlossm_mask_avg_b.val[0,k] / bsz, flowlossm_mask_avg_b.avg[0,k] / bsz,
                               deltaroterrm_mask.val[0,k] / bsz,    deltaroterrm_mask.avg[0,k] / bsz,
                             deltatranserrm_mask.val[0,k] / bsz,  deltatranserrm_mask.avg[0,k] / bsz,))

            ### Print stuff if we have weight sharpening enabled
            if args.use_wt_sharpening and not args.use_gt_masks:
                if args.use_gt_poses or args.decomp_model:
                    noise_std, pow = model.maskmodel.compute_wt_sharpening_stats(train_iter=num_train_iter)
                else:
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
                mode+'-dissimposeloss': dissimposeloss.data[0],
                mode+'-dissimdeltaloss': dissimdeltaloss.data[0],
                mode+'-deltarotloss': deltaroterrm.val,
                mode+'-deltatransloss': deltatranserrm.val,
            }
            for tag, value in info.items():
                tblogger.scalar_summary(tag, value, iterct)

            # (2) Log images & print predicted SE3s
            # TODO: Numpy or matplotlib
            if i % args.imgdisp_freq == 0:

                ## Log the images (at a lower rate for now)
                id = random.randint(0, sample['points'].size(0)-1)

                # Render the predicted and GT poses onto the depth
                # TODO: Assumes that we are using single jt data
                gtpose_1 = get_jt_poses(sample['poses'][id:id+1, 0], args.jt_ids).squeeze().clone()
                gtpose_2 = get_jt_poses(sample['poses'][id:id+1, 1], args.jt_ids).squeeze().clone()
                predpose_1, predpose_2, predpose_t_2 = pose_1[id].data.cpu(), pose_2[id].data.cpu(), pose_t_2[id].data.cpu()
                gtdepth_1 = normalize_img(sample['points'][id, 0, 2:].expand(3,args.img_ht,args.img_wd).permute(1,2,0), min=0, max=3)
                gtdepth_2 = normalize_img(sample['points'][id, 1, 2:].expand(3,args.img_ht,args.img_wd).permute(1,2,0), min=0, max=3)
                for k in xrange(args.num_se3):
                    # Pose_1 (GT/Pred)
                    util.draw_3d_frame(gtdepth_1, gtpose_1[k],     [0,0,1], args.cam_intrinsics, pixlength=15.0) # GT pose: Blue
                    util.draw_3d_frame(gtdepth_1, predpose_1[k],   [0,1,0], args.cam_intrinsics, pixlength=15.0) # Pred pose: Green
                    # Pose_1 (GT/Pred/Trans)
                    util.draw_3d_frame(gtdepth_2, gtpose_2[k],     [0,0,1], args.cam_intrinsics, pixlength=15.0) # GT pose: Blue
                    util.draw_3d_frame(gtdepth_2, predpose_2[k],   [0,1,0], args.cam_intrinsics, pixlength=15.0) # Pred pose: Green
                    util.draw_3d_frame(gtdepth_2, predpose_t_2[k], [1,0,0], args.cam_intrinsics, pixlength=15.0) # Transition model pred pose: Red
                depthdisp = torch.cat([gtdepth_1, gtdepth_2], 1).permute(2,0,1) # Concatenate along columns (3 x 240 x 640 image)

                # Concat the flows and masks into one tensor
                minf, maxf = min(fwdflows.data[id].min(), bwdflows.data[id].min()), \
                             max(fwdflows.data[id].max(), bwdflows.data[id].max())
                flowdisp  = torchvision.utils.make_grid(torch.cat([fwdflows.data.narrow(0,id,1),
                                                                   bwdflows.data.narrow(0,id,1),
                                                                   predfwdflows.data.narrow(0,id,1),
                                                                   predbwdflows.data.narrow(0,id,1)], 0).cpu(),
                                                        nrow=2, normalize=True, range=(minf, maxf))
                maskdisp  = torchvision.utils.make_grid(torch.cat([mask_1.data.narrow(0,id,1),
                                                                   mask_2.data.narrow(0,id,1)], 0).cpu().view(-1, 1, args.img_ht, args.img_wd),
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
                loss=lossm, fwdloss=ptlossm_f, bwdloss=ptlossm_b, consisloss=consislossm,
                flowloss_sum_f=flowlossm_sum_f, flowloss_sum_b=flowlossm_sum_b,
                flowloss_avg_f=flowlossm_avg_f, flowloss_avg_b=flowlossm_avg_b)
    print('\tPose-Dissim loss: {disP.val:.5f} ({disP.avg:.5f}), '
          'Delta-Dissim loss: {disD.val:.5f} ({disD.avg:.5f}), '
          'Delta-Rot-Err: {delR.val:.5f} ({delR.avg:.5f}), '
          'Delta-Trans-Err: {delt.val:.5f} ({delt.avg:.5f})'.format(
        disP=dissimposelossm, disD=dissimdeltalossm, delR=deltaroterrm, delt=deltatranserrm))
    print('========================================================')

    # Return the loss & flow loss
    return lossm, ptlossm_f, ptlossm_b, consislossm, \
           flowlossm_sum_f, flowlossm_avg_f, \
           flowlossm_sum_b, flowlossm_avg_b

### Print statistics
def print_stats(mode, epoch, curr, total, samplecurr, sampletotal,
                loss, fwdloss, bwdloss, consisloss,
                flowloss_sum_f, flowloss_avg_f,
                flowloss_sum_b, flowloss_avg_b):
    # Print loss
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], '
          'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
          'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
          'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
          'Consis: {consis.val:.3f} ({consis.avg:.3f})'.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, loss=loss, fwd=fwdloss, bwd=bwdloss,
        consis=consisloss))

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

### Compute pose/deltapose errors
# BOth inputs are tensors: B x S x K x 3 x 4, Output is a tensor: S (value is averaged over nSE3s and summed across the batch)
def compute_pose_errors(predposes, gtposes):
    batch, seq, nse3 = predposes.size(0), predposes.size(1), predposes.size(2)
    # Compute error per rotation matrix & translation vector
    roterr = (predposes.narrow(4,0,3) - gtposes.narrow(4,0,3)).pow(2).view(batch, seq, nse3, -1).sum(3).sqrt() # L2 of error in rotation (per matrix): B x S x K
    traerr = (predposes.narrow(4,3,1) - gtposes.narrow(4,3,1)).pow(2).view(batch, seq, nse3, -1).sum(3).sqrt() # L2 of error in translation (per vector): B x S x K
    return roterr.sum(2).sum(0) / nse3, traerr.sum(2).sum(0) / nse3

### Compute pose/deltapose errors (separate over channels)
# BOth inputs are tensors: B x S x K x 3 x 4, Output is a tensor: S x K
def compute_pose_errors_per_mask(predposes, gtposes):
    batch, seq, nse3 = predposes.size(0), predposes.size(1), predposes.size(2)
    # Compute error per rotation matrix & translation vector
    roterr = (predposes.narrow(4,0,3) - gtposes.narrow(4,0,3)).pow(2).view(batch, seq, nse3, -1).sum(3).sqrt() # L2 of error in rotation (per matrix): B x S x K
    traerr = (predposes.narrow(4,3,1) - gtposes.narrow(4,3,1)).pow(2).view(batch, seq, nse3, -1).sum(3).sqrt() # L2 of error in translation (per vector): B x S x K
    return roterr.sum(0), traerr.sum(0)

### Normalize image
def normalize_img(img, min=-0.01, max=0.01):
    return (img - min) / (max - min)

### Adjust learning rate
def adjust_learning_rate(optimizer, epoch, decay_rate=0.1, decay_epochs=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

### Convert continuous input into classes
def digitize(tensor, minv=-0.075, maxv=0.075, nbins=150):
    # Clamp tensor vals to min/max
    ctensor = tensor.clamp(min=minv, max=maxv)
    # Digitize it
    return torch.ceil(((ctensor - minv) / (maxv - minv)) * nbins).long() # Get ceil instead of round -> matches result from np.digitize or np.searchsorted

### Generate proper masks for the joints specified (all points between two joints have to belong the first joint & so on)
# The first channel of the masks is BG
# Masks is 4D: batchsz x seq x 1+njts x ht x wd
def get_jt_masks(masks, jt_ids):
    batch, seq, num, ht, wd = masks.size()
    jt_masks, jt_ids = torch.zeros(batch, seq, 1+len(jt_ids), ht, wd).type_as(masks), np.sort(jt_ids)+1
    for k in xrange(len(jt_ids)): # Add 1 as channel 0 is BG
        curr_jt = jt_ids[k]
        next_jt = num if (k == len(jt_ids)-1) else jt_ids[k+1] # Get next id (or go to end)
        jt_masks[:,:,k+1].copy_(masks[:,:,curr_jt:next_jt].sum(2)) # Basically all masks from current jt to next jt-1
    jt_masks[:,:,0].copy_(jt_masks[:,:,1:].sum(2)).eq_(0) # Bg mask
    return jt_masks # BG mask goes in front

### Get poses for the joints specified (add BG too)
# The first channel of the poses is BG
# Poses is 4D: batchsz x 1+njts x 3 x 4
def get_jt_poses(poses, jt_ids):
    jt_poses = [poses[:,0].unsqueeze(1)]  # Init with BG
    for id in np.sort(jt_ids)+1:
        jt_poses.append(poses[:,id].unsqueeze(1))
    return torch.cat(jt_poses, 1)


################ RUN MAIN
if __name__ == '__main__':
    main()
