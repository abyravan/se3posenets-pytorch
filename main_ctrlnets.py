# Global imports
import argparse
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

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
from util.tblogger import TBLogger

# Parse arguments
parser = argparse.ArgumentParser(description='SE3-Pose-Nets Training')

# Dataset options
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--train-per', default=0.6, type=float,
                    metavar='FRAC', help='fraction of data for the training set (default: 0.6)')
parser.add_argument('--val-per', default=0.15, type=float,
                    metavar='FRAC', help='fraction of data for the validation set (default: 0.15)')
parser.add_argument('--img-scale', default=1e-4, type=float,
                    metavar='IS', help='conversion scalar from depth resolution to meters (default: 1e-4)')
parser.add_argument('--step-len', default=1, type=int,
                    metavar='N', help='number of frames separating each example in the training sequence (default: 1)')
parser.add_argument('--seq-len', default=1, type=int,
                    metavar='N', help='length of the training sequence (default: 1)')
parser.add_argument('--ctrl-type', default='actdiffvel', type=str,
                    metavar='STR', help='Control type: actvel | actacc | comvel | comacc | comboth | [actdiffvel] | comdiffvel')

# Model options
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disables batch normalization (default: False)')
parser.add_argument('--nonlin', default='prelu', type=str,
                    metavar='NONLIN', help='type of non-linearity to use: [prelu] | relu | tanh | sigmoid | elu')
parser.add_argument('--se3-type', default='se3aa', type=str,
                    metavar='SE3', help='SE3 parameterization: [se3aa] | se3quat | se3spquat | se3euler | affine')
parser.add_argument('--pred-pivot', action='store_true', default=False,
                    help='Predict pivot in addition to the SE3 parameters (default: False)')
parser.add_argument('-n', '--num-se3', type=int, default=8,
                    help='Number of SE3s to predict (default: 8)')
parser.add_argument('--init-transse3-iden', action='store_true', default=False,
                    help='Initialize the weights for the SE3 prediction layer of the transition model to predict identity')

# Loss options
parser.add_argument('--fwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the FWD direction (default: 1)')
parser.add_argument('--bwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the BWD direction (default: 1)')
parser.add_argument('--consis-wt', default=0.01, type=float,
                    metavar='WT', help='Weight for the pose consistency loss (default: 0.01)')
parser.add_argument('--loss-scale', default=1000, type=float,
                    metavar='WT', help='Default scale factor for all the losses (default: 1000)')

# Training options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--train-ipe', default=2000, type=int, metavar='N',
                    help='number of training iterations per epoch (default: 1000)')
parser.add_argument('--val-ipe', default=500, type=int, metavar='N',
                    help='number of validation iterations per epoch (default: 500)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Optimization options
parser.add_argument('-o', '--optimization', default='adam', type=str,
                    metavar='OPTIM', help='type of optimization: sgd | [adam]')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr-decay', default=0.1, type=float, metavar='M',
                    help='Decay learning rate by this value every decay-epochs (default: 0.1)')
parser.add_argument('--decay-epochs', default=30, type=int,
                    metavar='M', help='Decay learning rate every this many epochs (default: 10)')

# Display/Save options
parser.add_argument('--disp-freq', '-p', default=10, type=int,
                    metavar='N', help='print/disp/save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--save-dir', default='results', type=str,
                    metavar='PATH', help='directory to save results in. If it doesnt exist, will be created. (default: results/)')

################ MAIN
def main():
    # Parse args
    global args
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()
    args.batch_norm = not args.no_batch_norm
    assert (args.seq_len == 1), "Recurrent network training not enabled currently"

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
    load_dir = args.data.split(',,')[0]
    args.baxter_labels = data.read_baxter_labels_file(load_dir + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']
    args.camera_data   = data.read_cameradata_file(load_dir + '/cameradata.txt')

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
    print('Predicting {} SE3s of type: {}'.format(args.num_se3, args.se3_type))

    # Camera parameters
    args.cam_intrinsics = {'fx': 589.3664541825391/2,
                           'fy': 589.3664541825391/2,
                           'cx': 320.5/2,
                           'cy': 240.5/2}
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => FWD: {}, BWD: {}, CONSIS: {}'.format(
        args.loss_scale, args.fwd_wt, args.bwd_wt, args.consis_wt))

    ### Create save directory and start tensorboard logger
    create_dir(args.save_dir) # Create directory
    now = time.strftime("%c")
    tblogger = TBLogger(args.save_dir + '/logs/' + now) # Start tensorboard logger

    # TODO: Add option for using encoder pose for tfm t2
    # TODO: Add options for mask sharpening approach
    # TODO: Add option for pre-conv BN + Nonlin

    ########################
    ############ Load datasets
    # Get datasets
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                       img_scale = args.img_scale, ctrl_type = 'actdiffvel',
                                                                       mesh_ids = args.mesh_ids, camera_data = args.camera_data)
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')   # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
    train_loader = DataEnumerator(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers, pin_memory=args.cuda))
    val_loader   = DataEnumerator(torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers, pin_memory=args.cuda))

    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset) # Run sequentially along the test dataset
    test_loader  = DataEnumerator(torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers, sampler = sampler,
                                        pin_memory=args.cuda))

    ########################
    ############ Load models & optimization stuff

    ### Load the model
    model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                  se3_type=args.se3_type, use_pivot=args.pred_pivot,
                                  use_kinchain=False, input_channels=3, use_bn=args.batch_norm,
                                  nonlinearity=args.nonlin, init_transse3_iden=args.init_transse3_iden)
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    ### Load optimizer
    optimizer = load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint       = torch.load(args.resume)
            loadargs         = checkpoint['args']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            assert (loadargs.optimization == args.optimization), "Optimizer in saved checkpoint ({}) does not match current argument ({})".format(
                    loadargs.optimization, args.optimization)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ########################
    ############ Test
    if args.evaluate:
        print('==== Evaluating pre-trained network on test data ===')
        iterate(test_loader, model, tblogger, len(test_loader), mode='test')

    ########################
    ############ Train / Validate
    best_val_loss, best_epoch = float("inf"), 0
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs)

        # Train for one epoch
        tr_loss, tr_fwdloss, tr_bwdloss, tr_consisloss,\
            tr_flowsum_f, tr_flowavg_f, \
            tr_flowsum_b, tr_flowavg_b = iterate(train_loader, model, tblogger, args.train_ipe,
                                                 mode='train', optimizer=optimizer, epoch=epoch+1)
        # Evaluate on validation set
        val_loss, val_fwdloss, val_bwdloss, val_consisloss, \
            val_flowsum_f, val_flowavg_f, \
            val_flowsum_b, val_flowavg_b = iterate(val_loader, model, tblogger, args.val_ipe,
                                                   mode='val', epoch=epoch+1)

        # Find best loss
        is_best       = (val_loss.avg < best_val_loss)
        prev_best_loss = best_val_loss
        if is_best:
            best_val_loss = val_loss.avg
            best_epoch    = epoch+1
            print('==== Epoch: {}, Improved on previous best loss ({}). Current: {} ===='.format(
                                    epoch+1, prev_best_loss, val_loss.avg))
        else:
            print('==== Epoch: {}, Did not improve on best loss ({}). Current: {} ===='.format(
                epoch + 1, prev_best_loss, val_loss.avg))
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch+1,
            'args' : args,
            'best_loss'  : best_val_loss,
            'train_stats': {'loss': tr_loss, 'fwdloss': tr_fwdloss,
                            'bwdloss': tr_bwdloss, 'consisloss': tr_consisloss,
                            'flowsum_f': tr_flowsum_f, 'flowavg_f': tr_flowavg_f,
                            'flowsum_b': tr_flowsum_b, 'flowavg_b': tr_flowavg_b,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count()
                            },
            'val_stats'  : {'loss': val_loss, 'fwdloss': val_fwdloss,
                            'bwdloss': val_bwdloss, 'consisloss': val_consisloss,
                            'flowsum_f': val_flowsum_f, 'flowavg_f': val_flowavg_f,
                            'flowsum_b': val_flowsum_b, 'flowavg_b': val_flowavg_b,
                            'niters': val_loader.niters, 'nruns': val_loader.nruns,
                            'totaliters': val_loader.iteration_count()
                            },
            'state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, is_best, savedir=args.save_dir)
        print('\n')

    # Do final testing (if not asked to evaluate)
    if not args.evaluate:
        print('==== Evaluating trained network on test data ====')
        iterate(test_loader, model, tblogger, len(test_loader), mode='test', epoch=args.epochs)
        print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                             best_epoch))

################# HELPER FUNCTIONS

### Main iterate function (train/test/val)
def iterate(data_loader, model, tblogger, num_iters,
            mode='test', optimizer=None, epoch=0):
    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    lossm, ptlossm_f, ptlossm_b, consislossm = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
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

    # Create a data-transformer (once per epoch)
    # TODO: Make this cleaner - put this into the data loader directly (collate_fn)
    # TODO: Batch along dim 1 instead of dim 0
    DataTransform = data.BaxterSeqDataTransformer(height=args.img_ht, width=args.img_wd,
                                                  intrinsics=args.cam_intrinsics,
                                                  meshids=args.mesh_ids)
    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(
        mode, epoch, num_iters))
    end = time.time()
    for i in xrange(num_iters):
        # ============ Load data ============#
        # Get a sample
        j, sample = data_loader.next()

        # Post-process sample
        if args.cuda:
            sample['points'], sample['masks'], sample['bwdflows'] \
                = DataTransform.process_sample(sample['depths'].cuda(), sample['labels'].cuda(),
                                               sample['poses'].cuda())
        else:
            sample['points'], sample['masks'], sample['bwdflows']\
                = DataTransform.process_sample(sample['depths'], sample['labels'],
                                               sample['poses'])

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        if args.cuda:
            pts_1    = to_var(sample['points'][:,0].clone(), requires_grad=True)
            pts_2    = to_var(sample['points'][:,1].clone(), requires_grad=True)
            ctrls_1  = to_var(sample['controls'][:,0].cuda(), requires_grad=True)
            fwdflows = to_var(sample['fwdflows'][:,0].cuda(), requires_grad=False)
            bwdflows = to_var(sample['bwdflows'][:,0].clone(), requires_grad=False)
        else:
            pts_1 = to_var(sample['points'][:, 0].clone(), requires_grad=True)
            pts_2 = to_var(sample['points'][:, 1].clone(), requires_grad=True)
            ctrls_1 = to_var(sample['controls'][:, 0].clone(), requires_grad=True)
            fwdflows = to_var(sample['fwdflows'][:, 0].clone(), requires_grad=False)
            bwdflows = to_var(sample['bwdflows'][:, 0].clone(), requires_grad=False)
        tarpts_1 = to_var(pts_1.data + fwdflows.data, requires_grad=False)
        tarpts_2 = to_var(pts_2.data + bwdflows.data, requires_grad=False)

        # Measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        # ============ FWD pass + Compute loss ============#
        # Run the FWD pass through the network
        [pose_1, mask_1], [pose_2, mask_2], [deltapose_t_12, pose_t_2] = model([pts_1, pts_2, ctrls_1])
        deltapose_t_21 = se3nn.RtInverse()(deltapose_t_12)  # Invert the delta-pose

        # Compute predicted points based on the masks
        # TODO: Add option for using encoder pose for tfm t2
        predpts_1 = se3nn.NTfm3D()(pts_1, mask_1, deltapose_t_12)
        predpts_2 = se3nn.NTfm3D()(pts_2, mask_2, deltapose_t_21)

        # Compute 3D point loss (3D losses @ t & t+1)
        # TODO: Switch based on soft mask vs wt sharpened mask
        # For soft mask model, compute losses without predicting points. Otherwise use predicted pts
        fwd_wt, bwd_wt, consis_wt = args.fwd_wt * args.loss_scale, args.bwd_wt * args.loss_scale, args.consis_wt * args.loss_scale
        if True:
            # Use the weighted 3D transform loss, do not use explicitly predicted points
            ptloss_1 = fwd_wt * se3nn.Weighted3DTransformLoss()(pts_1, mask_1, deltapose_t_12, tarpts_1)  # Predict pts in FWD dirn and compare to target @ t2
            ptloss_2 = bwd_wt * se3nn.Weighted3DTransformLoss()(pts_2, mask_2, deltapose_t_21, tarpts_2)  # Predict pts in BWD dirn and compare to target @ t1
        else:
            # Squared error between the predicted points and target points (Same as MSE loss)
            ptloss_1 = fwd_wt * ctrlnets.BiMSELoss(predpts_1, tarpts_1)
            ptloss_2 = bwd_wt * ctrlnets.BiMSELoss(predpts_2, tarpts_2)

        # Compute pose consistency loss
        consisloss = consis_wt * ctrlnets.BiMSELoss(pose_2, pose_t_2)  # Enforce consistency between pose @ t1 predicted by encoder & pose @ t1 from transition model

        # Compute total loss as sum of all losses
        loss = ptloss_1 + ptloss_2 + consisloss

        # Update stats
        ptlossm_f.update(ptloss_1.data[0]); ptlossm_b.update(ptloss_2.data[0])
        consislossm.update(consisloss.data[0]); lossm.update(loss.data[0])

        # Measure FWD time
        fwd_time.update(time.time() - end)
        end = time.time()

        # ============ Gradient backpass + Optimizer step ============#
        # Compute gradient and do optimizer update step (if in training mode)
        if (train):
            # Backward pass & optimize
            optimizer.zero_grad() # Zero gradients
            loss.backward()       # Compute gradients - BWD pass
            optimizer.step()      # Run update step

            # Measure BWD time
            bwd_time.update(time.time() - end)
            end = time.time()

        # ============ Visualization ============#
        # Compute flow predictions and errors
        predfwdflows = (predpts_1 - pts_1).float()
        predbwdflows = (predpts_2 - pts_2).float()
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
                        loss=lossm, fwdloss=ptlossm_f,
                        bwdloss=ptlossm_b, consisloss=consislossm,
                        flowloss_sum_f=flowlossm_sum_f, flowloss_sum_b=flowlossm_sum_b,
                        flowloss_avg_f=flowlossm_avg_f, flowloss_avg_b=flowlossm_avg_b)

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
                mode+'-consisloss': consisloss.data[0]
            }

            for tag, value in info.items():
                tblogger.scalar_summary(tag, value, iterct)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                tblogger.histo_summary(tag, to_np(value.data), iterct)
                tblogger.histo_summary(tag + '/grad', to_np(value.grad), iterct)

            ### Display images
            # TODO

        # Measure viz time
        viz_time.update(time.time() - end)
        end = time.time()

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=num_iters, total=num_iters,
                samplecurr=len(data_loader), sampletotal=len(data_loader),
                loss=lossm, fwdloss=ptlossm_f,
                bwdloss=ptlossm_b, consisloss=consislossm,
                flowloss_sum_f=flowlossm_sum_f, flowloss_sum_b=flowlossm_sum_b,
                flowloss_avg_f=flowlossm_avg_f, flowloss_avg_b=flowlossm_avg_b)
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
    print('Mode: {}, Epoch: {}, Iter: [{}/{}], Sample: [{}/{}], '
          'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
          'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
          'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
          'Consis: {consis.val:.3f} ({consis.avg:.3f})'.format(
        mode, epoch, curr, total, samplecurr, sampletotal, loss=loss, fwd=fwdloss,
        bwd=bwdloss, consis=consisloss))

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
            if (num_pts_d[j][0] > 0):
                loss_avg[k] += (loss_sum_d[j][0] / num_pts_d[j][0]) # Sum of per-point loss across the batch
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
                if (num_pts_d[i][0] > 0):
                    loss_avg[k][j]  += (loss_sum_d[i][0] / num_pts_d[i][0]) # Sum of per-point flow across batch
                    nz[k][j]        += 1 # One more dataset
    # Return
    return loss_sum, loss_avg, num_pts, nz

### Adjust learning rate
def adjust_learning_rate(optimizer, epoch, decay_rate=0.1, decay_epochs=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

### Check if torch variable is of type autograd.Variable
def is_var(x):
    return (type(x) == torch.autograd.variable.Variable)

### Convert variable to numpy array
def to_np(x):
    if is_var(x):
        return x.data.cpu().numpy()
    else:
        return x.cpu().numpy()

### Convert torch tensor to autograd.variable
def to_var(x, to_cuda=False, requires_grad=False):
    if torch.cuda.is_available() and to_cuda:
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

### Create a directory. From: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################# HELPER CLASSES

### Computes sum/avg stats
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

### Enumerate oer data
class DataEnumerator(object):
    """Allows iterating over a data loader easily"""
    def __init__(self, data):
        self.data   = data # Store the data
        self.len    = len(self.data) # Number of samples in the entire data
        self.niters = 0    # Num iterations in current run
        self.nruns  = 0    # Num rounds over the entire data
        self.enumerator = enumerate(self.data) # Keeps an iterator around

    def next(self):
        try:
            sample = self.enumerator.next() # Get next sample
        except StopIteration:
            self.enumerator = enumerate(self.data) # Reset enumerator once it reaches the end
            self.nruns += 1 # Done with one complete run of the data
            self.niters = 0 # Num iters in current run
            sample = self.enumerator.next() # Get next sample
            #print('Completed a run over the data. Num total runs: {}, Num total iters: {}'.format(
            #    self.nruns, self.niters+1))
        self.niters += 1 # Increment iteration count
        return sample # Return sample

    def __len__(self):
        return len(self.data)

    def iteration_count(self):
        return (self.nruns * self.len) + self.niters

################ RUN MAIN
if __name__ == '__main__':
    main()