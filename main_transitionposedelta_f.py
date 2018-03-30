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
import transnets
import util
from util import AverageMeter, Tee, DataEnumerator

#### Setup options
# Common
import argparse
import configargparse

def setup_comon_options():
    # Parse arguments
    parser = configargparse.ArgumentParser(description='Pose-Transition-Model Training')

    # Dataset options
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='Path to config file for parameters')
    parser.add_argument('-d', '--data-dir', default='', type=str, required=True,
                        metavar='DIR', help='path to folder containing train/test/val datasets')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--num-ctrl', default=7, type=int, metavar='N',
                        help='dimensionality of the control space (default: 7)')

    # Model options
    parser.add_argument('--no-batch-norm', action='store_true', default=False,
                        help='disables batch normalization (default: False)')
    parser.add_argument('--nonlin', default='prelu', type=str,
                        metavar='NONLIN', help='type of non-linearity to use: [prelu] | relu | tanh | sigmoid | elu | selu')
    parser.add_argument('--se3-type', default='se3aa', type=str,
                        metavar='SE3', help='SE3 parameterization: [se3aa] | se3quat | se3spquat | se3euler | affine | se3aar')
    parser.add_argument('-n', '--num-se3', type=int, default=8,
                        help='Number of SE3s to predict (default: 8)')
    parser.add_argument('--init-transse3-iden', action='store_true', default=False,
                        help='Initialize the weights for the SE3 prediction layer of the transition model to predict identity')
    parser.add_argument('--model-type', default='default', type=str,
                        help='Model type: [default] | simple | simplewide | dense | deep')

    # Loss options
    parser.add_argument('--loss-type', default='mse', type=str,
                        metavar='STR', help='Type of loss to use (default: mse | abs)')
    parser.add_argument('--loss-scale', default=10000, type=float,
                        metavar='WT', help='Default scale factor for all the losses (default: 1000)')

    # Training options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Optimization options
    parser.add_argument('-o', '--optimization', default='adam', type=str,
                        metavar='OPTIM', help='type of optimization: sgd | [adam]')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-decay', default=0.1, type=float, metavar='M',
                        help='Decay learning rate by this value every decay-epochs (default: 0.1)')
    parser.add_argument('--decay-epochs', default=30, type=int,
                        metavar='M', help='Decay learning rate every this many epochs (default: 10)')
    parser.add_argument('--min-lr', default=1e-5, type=float,
                        metavar='LR', help='min learning rate (default: 1e-5)')

    # Display/Save options
    parser.add_argument('--disp-freq', '-p', default=25, type=int,
                        metavar='N', help='print/disp/save frequency (default: 25)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-s', '--save-dir', default='results', type=str,
                        metavar='PATH', help='directory to save results in. If it doesnt exist, will be created. (default: results/)')

    # Return
    return parser


# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

# Saved pose checkpoint
def main():
    global args, num_train_iter
    parser = setup_comon_options()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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

    # Loss parameters
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    print('Loss scale: {}'.format(args.loss_scale))

    ### Load data from disk
    print("Loading training data from: {}".format(args.data_dir + "/transmodeldata_train.tar.gz"))
    data_tr = torch.load(args.data_dir + "/transmodeldata_train.tar.gz")
    predposes_1_tr, predposes_2_tr, gtposes_1_tr, gtposes_2_tr, ctrls_1_tr = \
        torch.cat(data_tr.predposes_1, 0).type(deftype), \
        torch.cat(data_tr.predposes_2, 0).type(deftype), \
        torch.cat(data_tr.gtposes_1, 0).type(deftype), \
        torch.cat(data_tr.gtposes_2, 0).type(deftype), \
        torch.cat(data_tr.ctrls_1, 0).type(deftype)

    print("Loading validation data from: {}".format(args.data_dir + "/transmodeldata_val.tar.gz"))
    data_vl = torch.load(args.data_dir + "/transmodeldata_val.tar.gz")
    predposes_1_vl, predposes_2_vl, gtposes_1_vl, gtposes_2_vl, ctrls_1_vl = \
        torch.cat(data_vl.predposes_1, 0).type(deftype), \
        torch.cat(data_vl.predposes_2, 0).type(deftype), \
        torch.cat(data_vl.gtposes_1, 0).type(deftype), \
        torch.cat(data_vl.gtposes_2, 0).type(deftype), \
        torch.cat(data_vl.ctrls_1, 0).type(deftype)

    print("Loading testing data from: {}".format(args.data_dir + "/transmodeldata_test.tar.gz"))
    data_te = torch.load(args.data_dir + "/transmodeldata_test.tar.gz")
    predposes_1_te, predposes_2_te, gtposes_1_te, gtposes_2_te, ctrls_1_te = \
        torch.cat(data_te.predposes_1, 0).type(deftype), \
        torch.cat(data_te.predposes_2, 0).type(deftype), \
        torch.cat(data_te.gtposes_1, 0).type(deftype), \
        torch.cat(data_te.gtposes_2, 0).type(deftype), \
        torch.cat(data_te.ctrls_1, 0).type(deftype)

    print("Num examples => train/val/test: {}/{}/{}".format(ctrls_1_tr.size(0)-1, ctrls_1_vl.size(0)-1, ctrls_1_te.size(0)-1))

    ######
    # Setup transition model
    num_train_iter = 0
    if args.model_type == 'default':
        modelfn = ctrlnets.TransitionModel
    elif args.model_type == 'simplewide':
        modelfn = lambda **v: transnets.SimpleTransitionModel(wide=True, **v)
    elif args.model_type == 'simple':
        modelfn = lambda **v: transnets.SimpleTransitionModel(wide=False, **v)
    elif args.model_type == 'deep':
        modelfn = transnets.DeepTransitionModel
    elif args.model_type == 'dense':
        modelfn = transnets.SimpleDenseNetTransitionModel
    else:
        assert False, "Unknown model type input: {}".format(args.model_type)

    # Load model
    model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                    delta_pivot='', se3_type=args.se3_type, use_kinchain=False,
                    nonlinearity=args.nonlin, init_se3_iden=args.init_transse3_iden,
                    local_delta_se3=False, use_jt_angles=False, num_state=0)
    if args.cuda:
        model.cuda()

    ### Load optimizer
    optimizer = load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    #### Resume from a checkpoint for transition model
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading transition model checkpoint '{}'".format(args.resume))
            checkpoint       = torch.load(args.resume)
            loadargs         = checkpoint['args']
            args.start_epoch = checkpoint['epoch']
            num_train_iter   = checkpoint['train_iter']
            try:
                model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
            except:
                model.load_state_dict(checkpoint['model_state_dict'])
            assert (loadargs.optimization == args.optimization), "Optimizer in saved checkpoint ({}) does not match current argument ({})".format(
                loadargs.optimization, args.optimization)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
                  .format(args.resume, checkpoint['epoch'], num_train_iter))
            best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
            best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
            print('==== Best validation loss: {} was from epoch: {} ===='.format(best_loss, best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_loss, best_epoch = float("inf"), 0

    ########################
    ############ Test (don't create the data loader unless needed, creates 4 extra threads)
    if args.evaluate:
        # Delete train and val loaders
        #del train_loader, val_loader

        # TODO: Move this to before the train/val loader creation??
        print('==== Evaluating pre-trained network on test data ===')
        test_stats = iterate(predposes_1_te, predposes_2_te, gtposes_1_te, gtposes_2_te, ctrls_1_te, model,
                             tblogger, mode='test', epoch=args.start_epoch+1)

        # Save final test error
        save_checkpoint({
            'args': args,
            'test_stats': test_stats,
        }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

        # Close log file & return
        logfile.close()
        return

    ## Create a file to log different validation errors over training epochs
    statstfile = open(args.save_dir + '/epochtrainstats.txt', 'w')
    statsvfile = open(args.save_dir + '/epochvalstats.txt', 'w')
    statstfile.write("Epoch, Loss, Consiserr, Consiserrmax\n")
    statsvfile.write("Epoch, Loss, Consiserr, Consiserrmax\n")

    ########################
    ############ Train / Validate
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs, args.min_lr)

        # Train for one epoch
        train_stats = iterate(predposes_1_tr, predposes_2_tr, gtposes_1_tr, gtposes_2_tr, ctrls_1_tr, model,
                              tblogger, mode='train', optimizer=optimizer, epoch=epoch+1)

        # Evaluate on validation set
        val_stats   = iterate(predposes_1_vl, predposes_2_vl, gtposes_1_vl, gtposes_2_vl, ctrls_1_vl, model,
                              tblogger, mode='val', epoch=epoch+1)

        # Find best losses
        val_loss = val_stats.loss.avg
        is_best = (val_loss < best_loss)
        prev_best_loss  = best_loss
        prev_best_epoch = best_epoch
        s ='SAME'
        if is_best:
            best_loss, best_epoch, s       = val_loss, epoch+1, 'IMPROVED'
        print('==== [LOSS]   Epoch: {}, Status: {}, Previous best: {:.5f}/{}. Current: {:.5f}/{} ===='.format(
                                    epoch+1, s, prev_best_loss, prev_best_epoch, best_loss, best_epoch))

        # Write losses to stats file
        statstfile.write("{}, {}, {}, {}\n".format(epoch+1, train_stats.loss.avg,
                                                   train_stats.consiserr.avg,
                                                   train_stats.consiserrmax.avg))
        statsvfile.write("{}, {}, {}, {}\n".format(epoch+1, val_stats.loss.avg,
                                                   val_stats.consiserr.avg,
                                                   val_stats.consiserrmax.avg))

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch+1,
            'args' : args,
            'best_loss'            : best_loss,
            'best_epoch'           : best_epoch,
            'train_stats'          : train_stats,
            'val_stats'            : val_stats,
            'train_iter'           : num_train_iter,
            'model_state_dict'     : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, is_best, savedir=args.save_dir, filename='checkpoint.pth.tar')
        print('\n')

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
    best_epoch   = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
    print('==== Best validation loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                             best_epoch))

    # Do final testing (if not asked to evaluate)
    # (don't create the data loader unless needed, creates 4 extra threads)
    print('==== Evaluating trained network on test data ====')
    test_stats = iterate(predposes_1_te, predposes_2_te, gtposes_1_te, gtposes_2_te, ctrls_1_te, model,
                         tblogger, mode='test', epoch=checkpoint['epoch'])
    print('==== Best validation loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                             best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'test_stats': test_stats,
    }, is_best=False, savedir=args.save_dir, filename='test_stats.pth.tar')

    # Write test stats to val stats file at the end
    statsvfile.write("{}, {}, {}, {}\n".format(checkpoint['epoch'], test_stats.loss.avg,
                                               test_stats.consiserr.avg,
                                               test_stats.consiserrmax.avg))
    statsvfile.close(); statstfile.close()

    # Close log file
    logfile.close()

################# HELPER FUNCTIONS

### Main iterate function (train/test/val)
def iterate(predposes_1, predposes_2, gtposes_1, gtposes_2, ctrls_1, model, tblogger,
            mode='test', optimizer=None, epoch=0):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Save all stats into a namespace
    stats = argparse.Namespace()
    stats.loss, stats.consiserrmax, stats.consiserr = AverageMeter(), AverageMeter(), AverageMeter()
    stats.data_ids = []

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

    # Create random sequence for the training/validation, test can be sequential
    nexamples = ctrls_1.size(0)-2
    deftype = 'torch.cuda.LongTensor' if args.cuda else 'torch.LongTensor' # Default tensor type
    if mode == 'train' or mode == 'val':
        stats.data_ids = torch.from_numpy(np.random.permutation(nexamples)).type(deftype) # shuffled list of integers from 0 -> ctrls.size(0)-2
    else:
        stats.data_ids = torch.Tensor([k for k in range(nexamples)]).type(deftype) # just normal list from 0 -> ctrls.size(0)-2
    nbatches = (nexamples + args.batch_size - 1) // args.batch_size

    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num batches: {} =========='.format(mode, epoch, nbatches))
    for i in xrange(nbatches):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get the ids of the data samples
        idvs = stats.data_ids[i*args.batch_size:min((i+1)*args.batch_size, nexamples)]
        pose_i = util.to_var(predposes_1[idvs], requires_grad=train) # input poses
        ctrl_i = util.to_var(ctrls_1[idvs], requires_grad=train)     # input ctrls
        pose_t = util.to_var(predposes_2[idvs], requires_grad=False) # target poses

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        # Run a fwd pass through the net
        delta_p, pose_p = model([pose_i, ctrl_i])

        # Compute loss
        if args.loss_type == 'mse':
            loss = args.loss_scale * ctrlnets.BiMSELoss(pose_p, pose_t)
        elif args.loss_type == 'abs':
            loss = args.loss_scale * ctrlnets.BiAbsLoss(pose_p, pose_t)
        else:
            assert False, "Unknown loss type: {}".format(args.loss_type)
        stats.loss.update(loss.data[0])

        # Measure FWD time
        fwd_time.update(time.time() - start)

        # ============ Gradient backpass + Optimizer step ============#
        # Compute gradient and do optimizer update step (if in training mode)
        if (train):
            # Start timer
            start = time.time()

            # Backward pass & optimize
            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Compute gradients - BWD pass
            optimizer.step()  # Run update step

            # Increment number of training iterations by 1
            num_train_iter += 1

            # Measure BWD time
            bwd_time.update(time.time() - start)

        # ============ Visualization ============#
        # Start timer
        start = time.time()

        ### Pose consistency error
        # Compute consistency error for display
        consiserr    = ctrlnets.BiAbsLoss(pose_p.data, pose_t.data)
        consiserrmax = (pose_p.data - pose_t.data).abs().max()
        stats.consiserr.update(consiserr)
        stats.consiserrmax.update(consiserrmax)

        # Display/Print frequency
        if i % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=i+1, total=nbatches, stats=stats)

            ### Print time taken
            print('\tTime => Data: {data.val:.3f} ({data.avg:.3f}), '
                        'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
                        'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
                        'Viz: {viz.val:.3f} ({viz.avg:.3f})'.format(
                    data=data_time, fwd=fwd_time, bwd=bwd_time, viz=viz_time))

            ### TensorBoard logging
            # (1) Log the scalar values
            iterct = epoch*nbatches + i # Get total number of iterations so far
            info = {
                mode+'-loss': loss.data[0],
                mode+'-consiserr': consiserr,
                mode+'-consiserrmax': consiserrmax,
            }
            if mode == 'train':
                info[mode+'-lr'] = args.curr_lr # Plot current learning rate
            for tag, value in info.items():
                tblogger.scalar_summary(tag, value, iterct)

            ## Print the predicted delta-SE3s
            if (i % (5*args.disp_freq)) == 0:
                id = np.random.randint(ctrl_i.size(0))
                deltase3s = predictions['deltase3'].data[id].view(args.num_se3, -1).cpu()
                print('\tPredicted delta-SE3s @ t=2:', deltase3s)

        # Measure viz time
        viz_time.update(time.time() - start)

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=nbatches, total=nbatches, stats=stats)
    print('========================================================')

    # Return the stats
    stats.data_ids = stats.data_ids.cpu()
    return stats

### Print statistics
def print_stats(mode, epoch, curr, total, stats):
    # Print loss
    print('Mode: {}, Epoch: [{:3}/{:3}], Iter: [{:5}/{:5}], '
          'Loss: {loss.val: 7.4f} ({loss.avg: 7.4f}),'
          'Consis: {cerr.val: 6.4f}/{cerrm.val: 6.4f} ({cerr.avg: 6.4f}/{cerrm.avg: 6.4f})'.format(
        mode, epoch, args.epochs, curr, total, loss=stats.loss,
        cerr=stats.consiserr, cerrm=stats.consiserrmax))

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
def save_checkpoint(state, is_best, savedir='.', filename='checkpoint.pth.tar'):
    savefile = savedir + '/' + filename
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, savedir + '/model_best.pth.tar')

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

################ RUN MAIN
if __name__ == '__main__':
    main()
