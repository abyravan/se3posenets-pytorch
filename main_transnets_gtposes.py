# Torch imports
import torch
import torch.nn as nn

# Global imports
import os
import sys
import shutil
import time
import numpy as np

# Local imports
import ctrlnets
import util
from util import Tee, AverageMeter

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

######################
#### Setup options
# Parse arguments
import argparse
import configargparse
parser = configargparse.ArgumentParser(description='Train transition models on GT pose data (Baxter)')

# Dataset options
parser.add_argument('-c', '--config', required=True, is_config_file=True,
                    help='Path to config file for parameters')
parser.add_argument('-d', '--data', required=True,
                    help='Path to tar file with pose data')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--step-len', default=1, type=int,
                    metavar='N', help='number of frames separating each example in the training sequence (default: 1)')
parser.add_argument('--seq-len', default=1, type=int,
                    metavar='N', help='length of the training sequence (default: 1)')

# Data options
parser.add_argument('--se3-type', default='se3aa', type=str, metavar='SE3',
                    help='SE3 parameterization: se3aa | se3quat')
parser.add_argument('--nonlin', default='prelu', type=str, metavar='NONLIN',
                    help='type of non-linearity to use: [prelu] | relu | tanh | sigmoid | elu')

# Transition model
parser.add_argument('--model', type=str, default='default', help='type of transition net model to train')
parser.add_argument('--init-se3-iden', action='store_true', default=False,
                    help='Initialize network predictions to identity (default: False)')
parser.add_argument('--use-kinchain', action='store_true', default=False,
                    help='Use kinematic chain structure in transition nets (default: False)')

# Loss options
parser.add_argument('--loss-type', default='mse', type=str, metavar='STR',
                    help='Type of loss to use (default: mse | abs)')
parser.add_argument('--loss-wt', default=1.0, type=float,
                    metavar='WT', help='Scale factor for the loss(default: 1)')

# Training options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--train-ipe', default=2000, type=int, metavar='N',
                    help='number of training iterations per epoch (default: 2000)')
parser.add_argument('--val-ipe', default=500, type=int, metavar='N',
                    help='number of validation iterations per epoch (default: 500)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Optimization options
parser.add_argument('-o', '--optimization', default='rmsprop', type=str,
                    metavar='OPTIM', help='type of optimization: sgd | adam | [rmsprop]')
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

# Display/Save options
parser.add_argument('--disp-freq', '-p', default=25, type=int,
                    metavar='N', help='print/disp/save frequency (default: 25)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--save-dir', default='results', type=str, metavar='PATH',
                    help='directory to save results in. If it doesnt exist, will be created. (default: results/)')

def dataset_size(dataset, step, seq):
    dataset_size = 0
    for k in xrange(len(dataset['gtposes'])):
        dataset_size += dataset['gtposes'][k].size(0) - step*(seq+1)
    return dataset_size

def dataset_hist(dataset, step ,seq):
    datahist = [0]
    for k in xrange(len(dataset['gtposes'])):
        numcurrdata = dataset['gtposes'][k].size(0) - step*(seq+1)
        datahist.append(datahist[-1] + numcurrdata)
    return datahist

def get_sample(dataset, idx, data_hist, step, seq):
    # Find which dataset to sample from
    numdata = data_hist[-1]
    assert (idx < numdata)  # Check if we are within limits
    did = np.digitize(idx, data_hist) - 1  # If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID

    # Find ID of sample in that dataset (not the same as idx as we might have multiple datasets)
    start = (idx - data_hist[did])  # This will be from 0 - size for either train/test/val part of that dataset
    end   = start + step*seq

    # Get sample
    dt = (1.0/30.0) * step
    poses    = dataset['gtposes'][did][start:end+1:step] # nseq+1 x nse3 x 3 x 4
    jtangles = dataset['jtangles'][did][start:end+1:step] # nseq+1 x 7
    ctrls    = (jtangles[1:] - jtangles[:-1]) / dt # nseq x 7
    return poses, jtangles, ctrls

######################
def main():
    ########################
    ## Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    ### Create save directory and start tensorboard logger
    util.create_dir(args.save_dir)  # Create directory
    now = time.strftime("%c")
    tblogger = util.TBLogger(args.save_dir + '/logs/' + now)  # Start tensorboard logger

    # Create logfile to save prints
    logfile = open(args.save_dir + '/logs/' + now + '/logfile.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    ########################
    ## Load data
    if args.data.find('~') != -1:
        args.data = os.path.expanduser(args.data)
    assert(os.path.exists(args.data))

    # Get the dataset
    dataset = torch.load(args.data)
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']
    assert(len(train_dataset['gtposes']) == len(test_dataset['gtposes']))
    assert(len(train_dataset['gtposes']) == len(val_dataset['gtposes']))
    train_size, val_size, test_size = dataset_size(train_dataset, args.step_len, args.seq_len),\
                                      dataset_size(val_dataset, args.step_len, args.seq_len),\
                                      dataset_size(test_dataset, args.step_len, args.seq_len)
    print('Dataset size => Train: {}, Val: {}, Test: {}'.format(train_size, val_size, test_size))

    # Set up some vars
    args.num_se3 = train_dataset['gtposes'][0].size(1)
    args.num_ctrl = train_dataset['jtangles'][0].size(-1)
    print('Num SE3: {}, Num ctrl: {}, SE3 Type: {}, Step/Sequence length: {}/{}'.format(
        args.num_se3, args.num_ctrl, args.se3_type, args.step_len, args.seq_len))

    ########################
    ## Check if datakey is valid
    ## Setup networks
    if args.model == 'default':
        model = ctrlnets.TransitionModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                         se3_type=args.se3_type, nonlinearity=args.nonlin,
                                         init_se3_iden=args.init_se3_iden, use_kinchain=args.use_kinchain,
                                         delta_pivot='', local_delta_se3=False, use_jt_angles=False)
    else:
        assert(False)
    if args.cuda:
        model.cuda()

    ### Load optimizer
    optimizer = load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            loadargs = checkpoint['args']
            load_dict = checkpoint['model_state_dict']
            model.load_state_dict(load_dict)
            assert (
            loadargs.optimization == args.optimization), "Optimizer in saved checkpoint ({}) does not match " \
                                                         "current argument ({})".format(
                                                          loadargs.optimization, args.optimization)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print('Loading optimizer failed, continuing with new optimizer')
            print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
                  .format(args.resume, checkpoint['epoch'], num_train_iter))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ########################
    # Train/Val
    args.iter_ctr = {'train': 0, 'val': 0, 'test': 0}
    best_val_loss, best_epoch, num_train_iter = float("inf"), 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs)

        # Train for one epoch
        train_stats = iterate(train_dataset, model, tblogger, args.train_ipe,
                              mode='train', optimizer=optimizer, epoch=epoch+1)

        # Evaluate on validation set
        val_stats = iterate(val_dataset, model, tblogger, args.val_ipe,
                            mode='val', epoch=epoch+1)

        # Find best loss
        val_loss = val_stats.loss  # all models have a flow-vel error - only from predicted/computed delta
        is_best = (val_loss.avg < best_val_loss)
        prev_best_loss = best_val_loss
        prev_best_epoch = best_epoch
        if is_best:
            best_val_loss = val_loss.avg
            best_epoch = epoch + 1
            print('==== Epoch: {}, Improved on previous best loss ({}) from epoch {}. Current: {} ===='.format(
                epoch + 1, prev_best_loss, prev_best_epoch, val_loss.avg))
        else:
            print('==== Epoch: {}, Did not improve on best loss ({}) from epoch {}. Current: {} ===='.format(
                epoch + 1, prev_best_loss, prev_best_epoch, val_loss.avg))

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'best_loss': best_val_loss,
            'best_epoch': best_epoch,
            'train_iter': num_train_iter,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best, savedir=args.save_dir, filename='checkpoint.pth.tar')#.format(epoch+1))
        print('\n')

    # Load best model for testing (not latest one)
    print("=> loading best model from '{}'".format(args.save_dir + "/model_best.pth.tar"))
    checkpoint = torch.load(args.save_dir + "/model_best.pth.tar")
    num_train_iter = checkpoint['train_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded best checkpoint (epoch {}, train iter {})".format(checkpoint['epoch'], num_train_iter))
    best_epoch = checkpoint['best_epoch']
    print('==== Best validation loss: {} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))

    # Do final testing (if not asked to evaluate)
    # (don't create the data loader unless needed, creates 4 extra threads)
    print('==== Evaluating trained network on test data ====')
    test_stats = iterate(test_dataset, model, tblogger, -1, mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'test_stats': test_stats
    }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

    # Close log file
    logfile.close()

###############
### Main iterate function (train/test/val)
def iterate(dataset, model, tblogger, num_iters, mode='test', optimizer=None, epoch=0):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    stats = argparse.Namespace()
    stats.loss, stats.seq_loss    = AverageMeter(), AverageMeter()
    stats.poseerrmean, stats.poseerrmax = AverageMeter(), AverageMeter()
    stats.poseerrindivmean, stats.poseerrindivmax = AverageMeter(), AverageMeter()

    # Switch model modes
    train = (mode == 'train')
    if train:
        assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
        model.train()
    else:
        assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}".format(mode)
        model.eval()

    # Run an epoch
    datahist = dataset_hist(dataset, args.step_len, args.seq_len)
    nexamples = datahist[-1]
    if (num_iters == -1):
        num_iters =  nexamples // args.batch_size
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(mode, epoch, num_iters))
    nseq = args.seq_len
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'  # Default tensor type
    assert(args.loss_type == 'mse' or args.loss_type == 'abs')
    losslayer = torch.nn.MSELoss() if args.loss_type == 'mse' else torch.nn.L1Loss()
    for jj in xrange(num_iters):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get the sample IDs (go in sequence for test)
        ## TODO: Can we make sure to cover whole dataset at train/val time?
        if mode == 'test':
            st_id, ed_id = jj * args.batch_size, (jj + 1) * args.batch_size
            ids = torch.from_numpy(np.arange(st_id, ed_id)).long()
        else:
            ids = torch.from_numpy(np.random.randint(0, nexamples, args.batch_size)).long()

        # Get the samples
        gtposes, jtangles, controls = [], [], []
        for k in xrange(ids.nelement()):
            poses_c, angles_c, ctrls_c = get_sample(dataset, ids[k], datahist, args.step_len, args.seq_len)
            gtposes.append(poses_c.unsqueeze(0))
            jtangles.append(angles_c.unsqueeze(0))
            controls.append(ctrls_c.unsqueeze(0))
        gtposes  = torch.cat(gtposes, 0).type(deftype) # B x (seq+1) x nse3 x 3 x 4
        jtangles = torch.cat(jtangles, 0).type(deftype) # B x (seq+1) x 7
        controls = torch.cat(controls, 0).type(deftype) # B x seq x 7

        # Get state, controls @ t = 0 & all other time steps (inputs)
        poses_0 = util.to_var(gtposes[:,0].contiguous(), requires_grad=True)
        poses_t = util.to_var(gtposes[:,1:].contiguous(), requires_grad=False)
        ctrls   = util.to_var(controls, requires_grad=True) # Ctrl @ t=0->T-1

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        # Run the FWD pass
        loss, seq_loss = 0., torch.zeros(args.seq_len)
        pred_poses, pred_deltas = [], []
        for k in xrange(args.seq_len):
            # Fwd pass
            inp = [poses_0, ctrls[:,0]] if (k == 0) else \
                  [pred_poses[-1], ctrls[:,k]] # Get inputs
            pred_delta, pred_pose = model(inp)

            # Compute loss
            curr_loss = args.loss_wt * losslayer(pred_pose, poses_t[:,k])
            pred_deltas.append(pred_delta) # Save delta
            pred_poses.append(pred_pose) # Save pose

            # Save stuff
            loss += curr_loss
            seq_loss[k] = curr_loss.data[0]

        # Save loss
        stats.loss.update(loss.data[0])
        stats.seq_loss.update(seq_loss)

        # Measure FWD time
        fwd_time.update(time.time() - start)
        args.iter_ctr[mode] += 1

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

        ### Pose error
        poseerrormean, poseerrormax = torch.zeros(args.seq_len), torch.zeros(args.seq_len)
        poseerrorindivmean, poseerrorindivmax = torch.zeros(args.seq_len,args.num_se3), \
                                                torch.zeros(args.seq_len,args.num_se3)
        for k in xrange(args.seq_len):
            diff = (pred_poses[k].data - poses_t[:,k].data).cpu().abs()
            poseerrormax[k]       = diff.max()
            poseerrormean[k]      = diff.mean()
            poseerrorindivmax[k]  = diff.permute(1,0,2,3).contiguous().view(args.num_se3,-1).max(dim=1)[0]
            poseerrorindivmean[k] = diff.permute(1,0,2,3).contiguous().view(args.num_se3,-1).mean(dim=1)
        stats.poseerrmean.update(poseerrormean)
        stats.poseerrmax.update(poseerrormax)
        stats.poseerrindivmean.update(poseerrorindivmean)
        stats.poseerrindivmax.update(poseerrorindivmax)

        ### Send errors to tensorboard
        # Display/Print frequency
        if jj % args.disp_freq == 0:
            ### Print statistics
            print_stats(mode, epoch=epoch, curr=jj+1, total=num_iters, stats=stats)

            ### Print time taken
            print('\tTime => Data: {data.val:.3f} ({data.avg:.3f}), '
                  'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
                  'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
                  'Viz: {viz.val:.3f} ({viz.avg:.3f})'.format(
                data=data_time, fwd=fwd_time, bwd=bwd_time, viz=viz_time))

            ### TensorBoard logging
            # (1) Log the scalar values
            iterct = args.iter_ctr[mode]  # Get total number of iterations so far
            info = {
                mode + '-loss': loss.data[0],
                mode + '-seqloss': seq_loss.sum(),
                mode + '-poseerrmean': poseerrormean.sum(),
                mode + '-poseerrmax': poseerrormax.sum(),
            }
            poseindivmean, poseindivmax = poseerrorindivmean.sum(0), poseerrorindivmax.sum(0)
            for kj in xrange(args.num_se3):
                info[mode + '-poseerrmean-{}'.format(kj)] = poseindivmean[kj]
                info[mode + '-poseerrmax-{}'.format(kj)]  = poseindivmax[kj]
            if mode == 'train':
                info[mode + '-lr'] = args.curr_lr  # Plot current learning rate
            for tag, value in info.items():
                tblogger.scalar_summary(tag, value, iterct)

        # Measure viz time
        viz_time.update(time.time() - start)

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=num_iters, total=num_iters, stats=stats)
    print('========================================================')

    # Return the loss & flow loss
    return stats

################
### Print statistics
def print_stats(mode, epoch, curr, total,  stats):
    # Print loss
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], '
          'Loss: {loss.val:.4f} ({loss.avg:.4f}), '.format(
        mode, epoch, args.epochs, curr, total, loss=stats.loss))

    # Print flow loss per timestep
    for k in xrange(args.seq_len):
        posestr = ', Pose: {:.3f}/{:.4f} ({:.3f}/{:.4f})'.format(
            stats.poseerrmean.val[k], stats.poseerrmean.avg[k],
            stats.poseerrmax.val[k], stats.poseerrmax.avg[k],
        )
        print('\tStep: {}{}'.format(1 + k, posestr))
        print(torch.cat([stats.poseerrindivmean.avg[k:k+1,:],
                         stats.poseerrindivmax.avg[k:k+1,:]], 0))

### Load optimizer
def load_optimizer(optim_type, parameters, lr=1e-3, momentum=0.9, weight_decay=1e-4):
    if optim_type == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr = lr, weight_decay= weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=parameters, lr=lr, momentum=momentum,
                                        weight_decay=weight_decay)
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
def adjust_learning_rate(optimizer, epoch, decay_rate=0.1, decay_epochs=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.curr_lr = lr

################ RUN MAIN
if __name__ == '__main__':
    main()
