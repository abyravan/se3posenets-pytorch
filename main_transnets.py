# Global imports
import os
import sys
import shutil
import time
import random

# Torch imports
import torch
import torch.optim
import torch.utils.data

# Local imports
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
parser.add_argument('--loss-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the delta loss (default: 0)')
parser.add_argument('--use-kinchain', action='store_true', default=False,
                        help='Kinematic chain assumption (default: False)')

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

    # Get dimensions of ctrl & state
    try:
        statelabels, ctrllabels = data.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
        print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
    except:
        statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
        ctrllabels = statelabels  # Just use the labels
        print("Could not read statectrllabels file. Reverting to labels in statelabels file")
    args.num_state, args.num_ctrl = len(statelabels), len(ctrllabels)
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
    print('Loss scale: {}, Delta loss wt => {}'.format(args.loss_scale, args.loss_wt))

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

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
                                                                       mesh_ids = args.mesh_ids, ctrl_ids=ctrlids_in_state,
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
    modelfn = se2nets.SE2TransitionModel if args.se2_data else ctrlnets.TransitionModel
    model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3, use_pivot=args.pred_pivot,
                    se3_type=args.se3_type, use_kinchain=args.use_kinchain, nonlinearity=args.nonlin,
                    init_se3_iden=args.init_transse3_iden, local_delta_se3=args.local_delta_se3)
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
        te_loss = iterate(test_loader, model, tblogger, len(test_loader), mode='test')

        # Save final test error
        save_checkpoint({
            'args': args,
            'test_stats': {'loss': te_loss,
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
        tr_loss, _, _ = iterate(train_loader, model, tblogger, args.train_ipe,
                          mode='train', optimizer=optimizer, epoch=epoch+1)

        # Log values and gradients of the parameters (histogram)
        # NOTE: Doing this in the loop makes the stats file super large / tensorboard processing slow
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            tblogger.histo_summary(tag, util.to_np(value.data), epoch + 1)
            tblogger.histo_summary(tag + '/grad', util.to_np(value.grad), epoch + 1)

        # Evaluate on validation set
        val_loss, _, _ = iterate(val_loader, model, tblogger, args.val_ipe,
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
            'train_stats': {'loss': tr_loss,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count()
                            },
            'val_stats'  : {'loss': val_loss,
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
    te_loss, _, _ = iterate(test_loader, model, tblogger, len(test_loader),
                     mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'best_loss': best_val_loss,
        'test_stats': {'loss': te_loss,
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
    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    lossm= AverageMeter()
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
    model.deltase3decoder.register_forward_hook(get_output('deltase3'))

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
        gtpose_1 = util.to_var(sample['poses'][:, 0].clone().type(deftype), requires_grad=train)  # 0 is BG, so we add 1
        gtpose_2 = util.to_var(sample['poses'][:, 1].clone().type(deftype), requires_grad=False)  # 0 is BG, so we add 1
        ctrl_1   = util.to_var(sample['controls'][:, 0].clone().type(deftype), requires_grad=train)
        gtdeltapose_t_12 = data.ComposeRtPair(gtpose_2, data.RtInverse(gtpose_1))  # Pose_t+1 * Pose_t^-1

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        # FWD pass
        deltapose_t_12, pose_2_t = model([gtpose_1, ctrl_1])

        ### Delta error
        delta_wt = args.loss_wt * args.loss_scale
        loss = delta_wt * ctrlnets.BiMSELoss(deltapose_t_12, gtdeltapose_t_12)

        # Update stats
        lossm.update(loss.data[0])

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

            # Measure BWD time
            bwd_time.update(time.time() - start)

        # ============ Visualization ============#
        # Start timer
        start = time.time()

        ### Pose error
        # Compute error in delta pose space (not used for backprop)
        deltaroterr, deltatranserr = compute_pose_errors(deltapose_t_12.data.unsqueeze(1).cpu(),
                                                         gtdeltapose_t_12.data.unsqueeze(1).cpu())
        deltaroterrm.update(deltaroterr[0]); deltatranserrm.update(deltatranserr[0])

        # Compute rot & trans err per pose channel
        if args.disp_err_per_mask:
            deltaroterr_mask, deltatranserr_mask = compute_pose_errors_per_mask(deltapose_t_12.data.unsqueeze(1).cpu(),
                                                                                gtdeltapose_t_12.data.unsqueeze(1).cpu())
            deltaroterrm_mask.update(deltaroterr_mask);  deltatranserrm_mask.update(deltatranserr_mask)

        ### Display/Print frequency
        if (i % args.disp_freq == 0) or (i == num_iters-1):
            # Print loss
            print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], '
                  'Loss: {loss.val:.4f} ({loss.avg:.4f}))'.format(
                mode, epoch, args.epochs, i, num_iters,
                data_loader.niters + 1, len(data_loader), loss=lossm))

            ### Print statistics
            print('\tDelta-Rot-Err: {delR.val:.5f} ({delR.avg:.5f}), '
                    'Delta-Trans-Err: {delt.val:.5f} ({delt.avg:.5f})'
                .format(delR=deltaroterrm, delt=deltatranserrm))

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
                mode+'-deltarotloss': deltaroterrm.val,
                mode+'-deltatransloss': deltatranserrm.val
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
                gtposea_1, gtposea_2 = gtpose_1[id].data.cpu(), gtpose_2[id].data.cpu()
                predposea_1, predposea_2 = gtpose_1[id].data.cpu(), pose_2_t[id].data.cpu()
                gtdepth_1 = normalize_img(sample['points'][id, 0, 2:].expand(3,args.img_ht,args.img_wd).permute(1,2,0), min=0, max=3)
                gtdepth_2 = normalize_img(sample['points'][id, 1, 2:].expand(3,args.img_ht,args.img_wd).permute(1,2,0), min=0, max=3)
                for k in xrange(args.num_se3):
                    # Pose_1 (GT/Pred)
                    util.draw_3d_frame(gtdepth_1, gtposea_1[k],     [0,0,1], args.cam_intrinsics, pixlength=15.0) # GT pose: Blue
                    util.draw_3d_frame(gtdepth_1, predposea_1[k],   [0,1,0], args.cam_intrinsics, pixlength=15.0) # Pred pose: Green
                    # Pose_1 (GT/Pred/Trans)
                    util.draw_3d_frame(gtdepth_2, gtposea_2[k],     [0,0,1], args.cam_intrinsics, pixlength=15.0) # GT pose: Blue
                    util.draw_3d_frame(gtdepth_2, predposea_2[k],   [0,1,0], args.cam_intrinsics, pixlength=15.0) # Pred pose: Green
                depthdisp = torch.cat([gtdepth_1, gtdepth_2], 1).permute(2,0,1) # Concatenate along columns (3 x 240 x 640 image)

                # Show as an image summary
                info = { mode+'-depths': util.to_np(depthdisp.unsqueeze(0))
                }
                for tag, images in info.items():
                    tblogger.image_summary(tag, images, iterct)

                ## Print the predicted delta-SE3s
                print '\tPredicted delta-SE3s @ t=2:', predictions['deltase3'].data[id].view(args.num_se3, args.se3_dim).cpu()

        # Measure viz time
        viz_time.update(time.time() - start)

    print('========================================================')

    # Return the loss & pose loss
    return lossm, deltaroterrm, deltatranserrm

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

################ RUN MAIN
if __name__ == '__main__':
    main()