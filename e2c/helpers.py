# Global imports
import configargparse
import shutil
import h5py

# Torch imports
import torch
import torch.distributions

# Local imports
import blockdata as BD
import e2c.model as e2cmodel

#####################################
############## OPTIONS FOR TRAINING NETS ON BLOCK STACKING DATA
def setup_common_options():
    # Parse arguments
    parser = configargparse.ArgumentParser(description='E2C/VAE training on block data')

    # Dataset options
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='Path to config file for parameters')
    parser.add_argument('-d', '--data', default=[], required=True,
                        action='append', metavar='DIRS', help='path to dataset(s), passed in as list [a,b,c...]')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-per', default=0.6, type=float,
                        metavar='FRAC', help='fraction of data for the training set (default: 0.6)')
    parser.add_argument('--val-per', default=0.15, type=float,
                        metavar='FRAC', help='fraction of data for the validation set (default: 0.15)')
    parser.add_argument('--step-len', default=1, type=int, metavar='N',
                        help='number of frames separating each example in the training sequence (default: 1)')
    parser.add_argument('--seq-len', default=1, type=int,
                        metavar='N', help='length of the training sequence (default: 1)')
    parser.add_argument('--ctrl-type', default='actdiffvel', type=str, metavar='STR',
                        help='Control type: actvel | [actdiffvel]')

    # Block dataset options
    parser.add_argument('--use-failures', action='store_true', default=False,
                        help='Use examples where the task planner fails. (default: False)')
    parser.add_argument('--robot', default='yumi', type=str, metavar='STR',
                        help='Robot to use: [yumi] | baxter')
    parser.add_argument('--gripper-ctrl-type', default='vel', type=str, metavar='STR',
                        help='Control specification for the gripper: [vel] | compos')

    # Encoder/Decoder options
    parser.add_argument('--enc-img-type', default='rgbd', type=str, metavar='EIMG',
                        help='Type of input image to the encoder: rgb | [rgbd] | rgbxyz | d | xyz')
    parser.add_argument('--dec-img-type', default='rgbd', type=str, metavar='EIMG',
                        help='Type of output image from the decoder: rgb | [rgbd] | rgbxyz | d | xyz')
    parser.add_argument('--enc-inp-state', action='store_true', default=False,
                        help='Use GT state as input to the encoder. (Default: False)')
    parser.add_argument('--dec-pred-state', action='store_true', default=False,
                        help='Predict state as output from the decoder. (Default: False)')
    parser.add_argument('--dec-pred-norm-rgb', action='store_true', default=False,
                        help='Normalize the predicted RGB data from the decoder to be from 0-1. (Default: False)')
    parser.add_argument('--conv-enc-dec', action='store_true', default=False,
                        help='Use a fully convolutional encoder/transition/decoder. (Default: False)')

    # Transition model options
    parser.add_argument('--trans-setting', default='dist2dist', type=str, metavar='TSET',
                        help='Input/Output of transition model: samp2samp | samp2dist | [dist2dist]')
    parser.add_argument('--trans-pred-deltas', action='store_true', default=False,
                        help='Transition model predicts deltas instead of next state directly. (Default: False)')
    parser.add_argument('--trans-model-type', default='nonlin', type=str,
                        help='Type of transition model: [nonlin] | loclin')

    # General options
    parser.add_argument('--norm-type', default='bn', type=str,
                        help='Type of norm to add to conv layers (none | [bn])')
    parser.add_argument('--nonlin-type', default='prelu', type=str, metavar='NONLIN',
                        help='type of non-linearity to use: [prelu] | relu | tanh | sigmoid | elu | selu')
    parser.add_argument('--wide-model', action='store_true', default=False,
                        help='Wider network')
    parser.add_argument('--coord-conv', action='store_true', default=False,
                        help='Use co-ordinate convolutions. (Default: False)')

    # Loss options
    parser.add_argument('--recons-loss-type', default='mse', type=str,
                       help='Reconstruction loss type: [mse] | abs )')
    parser.add_argument('--recons-wt', default=1.0, type=float,
                       metavar='WT', help='Weight for the reconstruction loss (default: 1.0)')
    parser.add_argument('--varkl-wt', default=1.0, type=float,
                       metavar='WT', help='Weight for the variational KL loss (default: 1.0)')
    parser.add_argument('--transenckl-wt', default=1.0, type=float,
                        metavar='WT', help='Weight for the KL consistency loss between transition '
                                           'model & encoder (default: 1.0)')
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
                        help='number of training iterations per epoch (default: 2000)')
    parser.add_argument('--val-ipe', default=500, type=int, metavar='N',
                        help='number of validation iterations per epoch (default: 500)')
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
    parser.add_argument('-s', '--save-dir', default='results', type=str, metavar='PATH',
                        help='directory to save results in. If it doesnt exist, will be created. (default: results/)')

    # Return
    return parser

#####################################
############## Block data loader

### Load block sequence from disk
def read_block_sequence_from_disk(dataset, id, ctrl_type='actdiffvel', robot='yumi',
                                  gripper_ctrl_type='vel'):
    # Setup vars
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics, camera_extrinsics = dataset['camera_intrinsics'], \
                                           dataset['camera_extrinsics']

    # Setup memory
    seq, path, fileid = BD.generate_block_sequence(dataset, id)  # Get the file paths

    ### Load data from the h5 file
    # Get depth, RGB, labels and poses
    with h5py.File(path, 'r') as h5data:
        ##### Read image data
        # Get RGB images
        rgbs = BD.NumpyBHWCToTorchBCHW(BD.ConvertPNGListToNumpy(h5data['images_rgb'][seq]))

        # Get depth images
        depths = BD.NumpyBHWToTorchBCHW(BD.ConvertDepthPNGListToNumpy(h5data['images_depth'][seq])).float()

        ##### Get joint state and controls
        dt = step_len * (1.0/30.0)
        if robot == 'yumi':
            # Indices for extracting current position with gripper, arm, etc. when
            # computing the state and controls for YUMI robot.
            arm_l_idx = [0, 2, 4, 6, 8, 10, 12, 15] # Last ID is gripper (left)
            arm_r_idx = [1, 3, 5, 7, 9, 11, 13, 14] # Last ID is gripper (right)

            # Get joint angles of right arm and gripper
            states   = torch.from_numpy(h5data['robot_positions'][seq][:, arm_r_idx]).float()
            if ctrl_type == 'actdiffvel':
                controls = (states[1:] - states[:-1]) / dt
            elif ctrl_type == 'actvel':
                # THIS IS NOT A GOOD IDEA AS I'VE SEEN CASES WHERE THE VELOCITIES ARE ZERO AT T BUT THERE IS A CHANGE
                # IN CONFIGURATION @ T + STEP (IF STEP IS LARGE ENOUGH)
                controls = torch.from_numpy(h5data['robot_positions'][seq[:-1]][:, arm_r_idx]).float()
            else:
                assert False, "Unknown control type input for the YUMI: {}".format(ctrl_type)

            # Gripper control
            if gripper_ctrl_type == 'vel':
                pass # This is what we have already
            elif gripper_ctrl_type == 'compos':
                ming, maxg = 0.015, 0.025 # Min/Max gripper positions
                gripper_cmds = (torch.from_numpy(h5data['right_gripper_cmd'][seq[:-1]]) - ming) / (maxg - ming)
                gripper_cmds.clamp_(0,1) # Normalize and clamp to 0/1
                controls[:,-1] = gripper_cmds # Update the gripper controls to be the actual position commands
            else:
                assert False, "Unknown gripper control type input for the YUMI: {}".format(gripper_ctrl_type)
        else:
            assert False, "Unknown robot type input: {}".format(robot)

        # Compute x & y values for the 3D points (= xygrid * depths)
        points = torch.zeros(seq_len+1, 3, camera_intrinsics['height'], camera_intrinsics['width'])
        xy = points[:, 0:2]
        xy.copy_(camera_intrinsics['xygrid'].expand_as(xy))  # = xygrid
        xy.mul_(depths.expand(seq_len + 1, 2, camera_intrinsics['height'], camera_intrinsics['width'])) # = xy * z
        points[:, 2:].copy_(depths) # Copy depths to 3D points

    # Return loaded data
    output = {'points': points, 'rgbs': rgbs, 'controls': controls,
              'states': states, 'dt': dt, 'fileid': int(fileid)}
    return output

### Setup the data loaders for the block datasets
def parse_options_and_setup_block_dataset_loader(args):
    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    #print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
    #    args.loss_scale, args.pt_wt, args.consis_wt))

    # Wide model
    if args.wide_model:
        print('Using a wider network')

    # YUMI robot
    if args.robot == "yumi":
        args.num_ctrl = 8
        args.img_ht, args.img_wd = 240, 320
        print("Img ht: {}, Img wd: {}, Num ctrl: {}".format(args.img_ht, args.img_wd, args.num_ctrl))
    else:
        assert False, "Unknown robot type input: {}".format(args.robot)

    ########################
    ### Load functions
    block_data = BD.read_block_sim_dataset(args.data,
                                           step_len=args.step_len,
                                           seq_len=args.seq_len,
                                           train_per=args.train_per,
                                           val_per=args.val_per,
                                           use_failures=args.use_failures)
    disk_read_func = lambda d, i: read_block_sequence_from_disk(d, i, ctrl_type=args.ctrl_type, robot=args.robot,
                                                                gripper_ctrl_type=args.gripper_ctrl_type)
    train_dataset = BD.BlockSeqDataset(block_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = BD.BlockSeqDataset(block_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = BD.BlockSeqDataset(block_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset),
                                                                       len(val_dataset),
                                                                       len(test_dataset)))

    # Return
    return train_dataset, val_dataset, test_dataset

# Setup image data, pts: B x (S+1) x C x H x W
def concat_image_data(pts, rgbs, img_type):
    if img_type == 'rgb':
        return rgbs
    elif img_type == 'xyz':
        return pts
    elif img_type == 'd':
        return pts.narrow(2,2,1).contiguous() # Get the depth channel
    elif img_type == 'rgbd':
        return torch.cat([rgbs, pts.narrow(2,2,1)], 2) # Concat along channels dimension
    elif img_type == 'rgbxyz':
        return torch.cat([rgbs, pts], 2) # Concat along channels dimension
    else:
        assert False, "Unknown image type input: {}".format(img_type)

# Split image data into rgbs, depths and xyz
# Return rgbs, depths, xyz points
def split_image_data(imgs, img_type, split_dim=2):
    if img_type == 'rgb':
        return imgs, None, None
    elif img_type == 'xyz':
        return None, None, imgs
    elif img_type == 'd':
        return None, imgs, None
    elif img_type == 'rgbd':
        return imgs.narrow(split_dim,0,3).contiguous(), \
               imgs.narrow(split_dim,3,1).contiguous(), None # Split along channels dimension
    elif img_type == 'rgbxyz':
        return imgs.narrow(split_dim,0,3).contiguous(), \
               imgs.narrow(split_dim,5,1).contiguous(), \
               imgs.narrow(split_dim,3,3).contiguous()  # Split along channels dimension
    else:
        assert False, "Unknown image type input: {}".format(img_type)

#####################################
############## Loss functions
# Default reconstruction loss function
def get_loss_function(loss_func_type, size_average=True):
    if loss_func_type == 'mse':
        return torch.nn.MSELoss(size_average=size_average)
    elif loss_func_type == 'abs':
        return torch.nn.L1Loss(size_average=size_average)
    else:
        assert False, "Unknown loss type: {}".format(loss_func_type)

# Variational KL loss, loss between the predicted encoder distribution and
# a 0 mean, 1-std deviation multivariate normal distribution
def variational_normal_kl_loss(p, size_average=True):
    # Create a 0-mean, 1-std deviation distribution
    bsz, ndim = p.mean.size() # 2D
    if isinstance(p, e2cmodel.MVNormal):
        q = e2cmodel.MVNormal(loc=torch.zeros(bsz,ndim).type_as(p.mean),
                              covariance_matrix=torch.eye(ndim).unsqueeze(0).repeat(bsz,1,1).type_as(p.mean))
    elif isinstance(p, e2cmodel.Normal):
        q = e2cmodel.Normal(loc=torch.zeros(bsz,ndim).type_as(p.mean),
                            scale=torch.ones(bsz,ndim).type_as(p.mean))
    else:
        assert False, 'Input distribution p is instance of class: {}'.format(type(p))

    # Return KL between p and q
    kl_loss = torch.distributions.kl.kl_divergence(p, q)
    if size_average:
        return kl_loss.mean()
    else:
        return kl_loss.sum()

#####################################
############## Other helpers

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

### Normalize image
def normalize_img(img, min=-0.01, max=0.01):
    return (img - min) / (max - min)

### Adjust learning rate
def adjust_learning_rate(optimizer, args, epoch, decay_rate=0.1, decay_epochs=10, min_lr=1e-5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    lr = min_lr if (args.lr < min_lr) else lr  # Clamp at min_lr
    print("======== Epoch: {}, Initial learning rate: {}, Current: {}, Min: {} =========".format(
        epoch, args.lr, lr, min_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.curr_lr = lr

