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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--tp', '--train-per', default=0.6, type=float,
                    metavar='FRAC', help='fraction of data to use for the training set')
parser.add_argument('--vp', '--val-per', default=0.15, type=float,
                    metavar='FRAC', help='fraction of data to use for the validation set')
parser.add_argument('--is', '--img-scale', default=1e-4, type=float,
                    metavar='IS', help='conversion scalar from depth resolution to meters')
parser.add_argument('--spl', '--step-len', default=1, type=int,
                    metavar='N', help='number of frames separating each example in the training sequence')
parser.add_argument('--sel', '--seq-len', default=1, type=int,
                    metavar='N', help='length of the training sequence')
parser.add_argument('--ct', '--ctrl-type', default='actdiffvel', type=str,
                    metavar='STR', help='Control type: actvel | actacc | comvel | comacc | comboth | actdiffvel | comdiffvel')

# Model options
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disables batch normalization')
parser.add_argument('--nl', '--nonlin', default='prelu', type=str,
                    metavar='NONLIN', help='type of non-linearity to use: prelu | relu | tanh | sigmoid | elu')
parser.add_argument('--se3', '--se3-type', default='se3aa', type=str,
                    metavar='SE3', help='SE3 parameterization: se3aa | se3quat | se3spquat | se3euler | affine')
parser.add_argument('--pv', '--pred-pivot', action='store_true', default=False,
                    help='Predict pivot in addition to the SE3 parameters')
parser.add_argument('-n', '--num-se3', type=int, default=8,
                    help='Number of SE3s to predict')

# Loss options
parser.add_argument('--fwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the FWD direction')
parser.add_argument('--bwd-wt', default=1.0, type=float,
                    metavar='WT', help='Weight for the 3D point based loss in the BWD direction')
parser.add_argument('--consis-wt', default=0.01, type=float,
                    metavar='WT', help='Weight for the pose consistency loss')


# Training options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Optimization options
parser.add_argument('-o', '--optimization', default='adam', type=str,
                    metavar='OPTIM', help='type of optimization: sgd | adam')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Display/Save options
parser.add_argument('--disp-freq', '-p', default=10, type=int,
                    metavar='N', help='print/disp/save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--save-dir', default='results', type=str,
                    metavar='PATH', help='directory to save results in. If it doesnt exist, will be created.')

################ MAIN
def main():
    # Parse args
    global args
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()
    args.batch_norm = not args.no_batch_norm

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup extra options
    args.img_ht, args.img_wd, args.img_suffix = 240, 320, 'sub'
    args.num_ctrl = 14 if args.ctrl_type.find('both') else 7 # Number of control dimensions
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
    args._fx, args._fy, args._cx, args._cy = 589.3664541825391/2, 589.3664541825391/2, 320.5/2, 240.5/2
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss weights => FWD: {}, BWD: {}, CONSIS: {}'.format(args.fwd_wt, args.bwd_wt, args.consis_wt))

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, pin_memory=args.cuda)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, pin_memory=args.cuda)

    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset) # Run sequentially along the test dataset
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers, sampler = sampler, pin_memory=args.cuda)

    ########################
    ############ Load models & optimization stuff

    ### Load the model
    model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                  se3_type=args.se3_type, use_pivot=args.pred_pivot,
                                  use_kinchain=False, input_channels=3, use_bn=args.batch_norm,
                                  nonlinearity=args.nonlin)
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    ### Load optimizer
    optimizer = load_optimizer(args.optimization, model.parameters(), lr=args.learning_rate,
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
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ########################
    ############ Test if asked to just test
    if args.evaluate:
        iterate(test_loader, model=model, train=False)


################# HELPER FUNCTIONS
### Main iterate function (train/test/val)
def iterate(data_loader, model, train=False, optimizer=None):
    # Setup avg stats:
    batchfwd_time = AverageMeter()
    batchbwd_time = AverageMeter()
    data_time     = AverageMeter()

    # Switch model modes
    if train:
        model.train()
    else:
        model.eval()

    # Run an epoch
    end = time.time()
    for i, sample in enumerate(data_loader):
        ## Get data

        # Measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        ## Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Measure FWD time
        batchfwd_time.update(time.time() - end)
        end = time.time()

        ## Compute gradient and do optimizer update step
        optimizer.zero_grad() # Zero gradients
        loss.backward()       # Compute gradients - BWD pass
        optimizer.step()      # Run update step

        # Measure BWD time
        batchbwd_time.update(time.time() - end)
        end = time.time()

        # Display/Print frequency
        if i % args.disp_freq == 0:
            print("Loss: {}", 0.1)

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
### TODO: Update function to save copies correctly
def save_checkpoint(state, copy, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if copy:
        shutil.copyfile(filename, 'model_curr.pth.tar')

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

### Adjust learning rate
### TODO: Update function to take argument for lr decay
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

################ RUN MAIN
if __name__ == '__main__':
    main()



