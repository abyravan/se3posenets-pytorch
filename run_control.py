# Global imports
import _init_paths
import argparse
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# TODO: Make this cleaner, we don't need most of these parameters to
# create the pangolin window
img_ht, img_wd, img_scale = 240, 320, 1e-4
seq_len = 1  # For now, only single step
num_se3 = 8  # TODO: Especially this parameter!
dt = 1.0 / 30.0
oldgrippermodel = False  # TODO: When are we actually going to use the new ones?
cam_intrinsics = {'fx': 589.3664541825391 / 2,
                  'fy': 589.3664541825391 / 2,
                  'cx': 320.5 / 2,
                  'cy': 240.5 / 2}
savedir = 'temp'  # TODO: Fix this!

# Load pangolin visualizer library
from torchviz import pangoviz
pangolin = pangoviz.PyPangolinViz(seq_len, img_ht, img_wd, img_scale, num_se3,
                                  cam_intrinsics['fx'], cam_intrinsics['fy'],
                                  cam_intrinsics['cx'], cam_intrinsics['cy'],
                                  dt, oldgrippermodel, savedir)

##########
# NOTE: When importing torch before initializing the pangolin window, I get the error:
#   Pangolin X11: Unable to retrieve framebuffer options
# Long story short, initializing torch before pangolin messes things up big time.
# Also, only the conda version of torch works otherwise there's an issue with loading the torchviz library before torch
#   ImportError: dlopen: cannot load any more object with static TLS
# With the new CUDA & NVIDIA drivers the new conda also doesn't work. had
# to move to CYTHON to get code to work

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import util
from util import AverageMeter

##########
# Parse arguments
parser = argparse.ArgumentParser(
    description='Reactive control using SE3-Pose-Nets')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to saved network to use for training (default: none)')

# Problem options
parser.add_argument('--start-id', default=-1, type=int, metavar='N',
                    help='ID in the test dataset for the start state (default: -1 = randomly chosen)')
parser.add_argument('--goal-horizon', default=1.5, type=float, metavar='SEC',
                    help='Planning goal horizon in seconds (default: 1.5)')
parser.add_argument('--only-top4-jts', action='store_true', default=False,
                    help='Controlling only the first 4 joints (default: False)')

# Planner options
parser.add_argument('--optimization', default='gn', type=str, metavar='OPTIM',
                    help='Type of optimizer to use: [gn] | backprop')
parser.add_argument('--max-iter', default=100, type=int, metavar='N',
                    help='Maximum number of planning iterations (default: 100)')
parser.add_argument('--gn-perturb', default=1e-3, type=float, metavar='EPS',
                    help='Perturbation for the finite-differencing to compute the jacobian (default: 1e-3)')
parser.add_argument('--gn-lambda', default=1e-4, type=float, metavar='LAMBDA',
                    help='Damping constant (default: 1e-4)')
parser.add_argument('--gn-jac-check', action='store_true', default=False,
                    help='check FD jacobian & gradient against the numerical jacobian & backprop gradient (default: False)')
parser.add_argument('--max-ctrl-mag', default=1.0, type=float, metavar='UMAX',
                    help='Maximum allowable control magnitude (default: 1 rad/s)')
parser.add_argument('--ctrl-mag-decay', default=0.99, type=float, metavar='W',
                    help='Decay the control magnitude by scaling by this weight after each iter (default: 0.99)')
parser.add_argument('--loss-scale', default=1000, type=float, metavar='WT',
                    help='Scaling factor for the loss (default: 1000)')

# TODO: Add criteria for convergence

# Misc options
parser.add_argument('--disp-freq', '-p', default=20, type=int,
                    metavar='N', help='print/disp/save frequency (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Display/Save options
parser.add_argument('-s', '--save-dir', default='', type=str,
                    metavar='PATH', help='directory to save results in. (default: <checkpoint_dir>/planlogs/)')


def main():
    # Parse args
    global pargs, args, num_train_iter
    pargs = parser.parse_args()
    pargs.cuda = not pargs.no_cuda and torch.cuda.is_available()

    # Create save directory and start tensorboard logger
    if pargs.save_dir == '':
        checkpoint_dir = pargs.checkpoint.rpartition('/')[0]
        pargs.save_dir = checkpoint_dir + '/planlogs/'
    print('Saving planning logs at: ' + pargs.save_dir)
    util.create_dir(pargs.save_dir)  # Create directory
    # Start tensorboard logger
    tblogger = util.TBLogger(pargs.save_dir + '/planlogs/')

    # Set seed
    torch.manual_seed(pargs.seed)
    if pargs.cuda:
        torch.cuda.manual_seed(pargs.seed)

    # Default tensor type
    # Default tensor type
    deftype = 'torch.cuda.FloatTensor' if pargs.cuda else 'torch.FloatTensor'

    # Invert a matrix to initialize the torch inverse code
    temp = torch.rand(7, 7).type(deftype)
    tempinv = torch.inverse(temp)

    ########################
    # Load pre-trained network

    # Load data from checkpoint
    # TODO: Print some stats on the training so far, reset best validation
    # loss, best epoch etc
    if os.path.isfile(pargs.checkpoint):
        print("=> loading checkpoint '{}'".format(pargs.checkpoint))
        checkpoint = torch.load(pargs.checkpoint)
        args = checkpoint['args']
        try:
            num_train_iter = checkpoint['num_train_iter']
        except:
            num_train_iter = checkpoint['epoch'] * args.train_ipe
        print("=> loaded checkpoint (epoch: {}, num train iter: {})"
              .format(checkpoint['epoch'], num_train_iter))
    else:
        print("=> no checkpoint found at '{}'".format(pargs.checkpoint))
        raise RuntimeError

    # Create a model
    if args.seq_len == 1:
        model = ctrlnets.SE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                      se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                      input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                      init_posese3_iden=False, init_transse3_iden=False,
                                      use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                                      sharpen_rate=args.sharpen_rate, pre_conv=False, wide=args.wide_model)  # TODO: pre-conv
        posemaskpredfn = model.posemaskmodel.forward
    else:
        model = ctrlnets.MultiStepSE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                               se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                               input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                               init_posese3_iden=args.init_posese3_iden,
                                               init_transse3_iden=args.init_transse3_iden,
                                               use_wt_sharpening=args.use_wt_sharpening,
                                               sharpen_start_iter=args.sharpen_start_iter,
                                               sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                                               decomp_model=args.decomp_model, wide=args.wide_model)
        posemaskpredfn = model.forward_pose_mask
    if pargs.cuda:
        model.cuda()  # Convert to CUDA if enabled

    # Update parameters from trained network
    try:
        # BWDs compatibility (TODO: remove)
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluate mode
    model.eval()

    # Sanity check some parameters (TODO: remove it later)
    assert(args.num_se3 == num_se3)
    assert(args.img_scale == img_scale)
    try:
        cam_i = args.cam_intrinsics
        for _, key in enumerate(cam_intrinsics):
            assert(cam_intrinsics[key] == cam_i[key])
    except AttributeError:
        args.cam_intrinsics = cam_intrinsics  # In case it doesn't exist

    ########################
    # Get the data
    # Get datasets (TODO: Make this path variable)
    data_path = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/'
    args.cam_extrinsics = data.read_cameradata_file(
        data_path + '/cameradata.txt')  # TODO: BWDs compatibility
    args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               args.cam_intrinsics)  # TODO: BWDs compatibility
    baxter_data = data.read_recurrent_baxter_dataset(data_path, args.img_suffix,
                                                     step_len=1, seq_len=1,
                                                     train_per=args.train_per, val_per=args.val_per)

    def disk_read_func(d, i): return data.read_baxter_sequence_from_disk(d, i, img_ht=args.img_ht, img_wd=args.img_wd,
                                                                         img_scale=args.img_scale,
                                                                         ctrl_type='actdiffvel',
                                                                         mesh_ids=args.mesh_ids,
                                                                         camera_extrinsics=args.cam_extrinsics,
                                                                         camera_intrinsics=args.cam_intrinsics)
    test_dataset = data.BaxterSeqDataset(
        baxter_data, disk_read_func, 'test')  # Test dataset

    # Get start & goal samples
    start_id = pargs.start_id if (
        pargs.start_id >= 0) else np.random.randint(len(test_dataset))
    goal_id = start_id + round(pargs.goal_horizon / dt)
    print('Test dataset size: {}, Start ID: {}, Goal ID: {}, Duration: {}'.format(len(test_dataset),
                                                                                  start_id, goal_id, pargs.goal_horizon))
    start_sample = test_dataset[start_id]
    goal_sample = test_dataset[goal_id]

    # Get the joint angles
    start_angles = start_sample['actconfigs'][0]
    goal_angles = goal_sample['actconfigs'][0]
    if pargs.only_top4_jts:
        print('Controlling only top 4 joints')
        goal_angles[4:] = start_angles[4:]

    ########################
    # Get start & goal point clouds, predict poses & masks
    # Initialize problem
    start_pts, da_goal_pts = torch.zeros(
        1, 3, args.img_ht, args.img_wd), torch.zeros(1, 3, args.img_ht, args.img_wd)
    pangolin.init_problem(start_angles.numpy(), goal_angles.numpy(
    ), start_pts[0].numpy(), da_goal_pts[0].numpy())

    # Get full goal point cloud
    goal_pts = generate_ptcloud(goal_angles)

    # Predict start/goal poses and masks
    print('Predicting start/goal poses and masks')
    start_poses, start_masks = posemaskpredfn(util.to_var(
        start_pts.type(deftype)), train_iter=num_train_iter)
    goal_poses, goal_masks = posemaskpredfn(util.to_var(
        goal_pts.type(deftype)), train_iter=num_train_iter)

    # Display the masks as an image summary
    maskdisp = torchvision.utils.make_grid(torch.cat([start_masks.data, goal_masks.data],
                                                     0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                                           nrow=args.num_se3, normalize=True, range=(0, 1))
    info = {'start/goal masks': util.to_np(maskdisp.narrow(0, 0, 1))}
    for tag, images in info.items():
        tblogger.image_summary(tag, images, 0)

    # Render the poses
    # NOTE: Data passed into cpp library needs to be assigned to specific
    # vars, not created on the fly (else gc will free it)
    start_poses_f, goal_poses_f = start_poses.data.cpu(
    ).float(), goal_poses.data.cpu().float()
    pangolin.initialize_poses(
        start_poses_f[0].numpy(), goal_poses_f[0].numpy())

    # Print error
    print('Initial jt angle error:')
    full_deg_error = (start_angles - goal_angles) * \
        (180.0 / np.pi)  # Full error in degrees
    print(full_deg_error.view(7, 1))

    ########################
    # Run the controller
    # Init stuff
    ctrl_mag = pargs.max_ctrl_mag
    angles, deg_errors = torch.FloatTensor(
        pargs.max_iter + 1, 7), torch.FloatTensor(pargs.max_iter + 1, 7)
    angles[0], deg_errors[0] = start_angles, full_deg_error
    ctrl_grads, ctrls = torch.FloatTensor(
        pargs.max_iter, args.num_ctrl), torch.FloatTensor(pargs.max_iter, args.num_ctrl)
    losses = torch.FloatTensor(pargs.max_iter)

    # Init vars for all items
    init_ctrl_v = util.to_var(torch.zeros(1, args.num_ctrl).type(
        deftype), requires_grad=True)  # Need grad w.r.t this
    goal_poses_v = util.to_var(goal_poses.data, requires_grad=False)

    # Plots for errors and loss
    fig, axes = plt.subplots(2, 1)
    fig.show()

    # Run the controller
    gen_time, posemask_time, optim_time, viz_time, rest_time = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for it in xrange(pargs.max_iter):
        # Print
        print('\n #####################')

        # Get current point cloud
        start = time.time()
        curr_angles = angles[it]
        curr_pts = generate_ptcloud(curr_angles).type(deftype)
        gen_time.update(time.time() - start)

        # Predict poses and masks
        start = time.time()
        curr_poses, curr_masks = posemaskpredfn(
            util.to_var(curr_pts), train_iter=num_train_iter)
        curr_poses_f, curr_masks_f = curr_poses.data.cpu(
        ).float(), curr_masks.data.cpu().float()
        posemask_time.update(time.time() - start)

        # Render poses and masks using Pangolin
        start = time.time()
        _, curr_labels = curr_masks_f.max(dim=1)
        curr_labels_f = curr_labels.float()
        pangolin.update_masklabels_and_poses(
            curr_labels_f.numpy(), curr_poses_f[0].numpy())

        # Show masks using tensor flow
        if (it % args.disp_freq) == 0:
            maskdisp = torchvision.utils.make_grid(curr_masks.data.cpu().view(-1, 1, args.img_ht, args.img_wd),
                                                   nrow=args.num_se3, normalize=True, range=(0, 1))
            info = {'curr masks': util.to_np(maskdisp.narrow(0, 0, 1))}
            for tag, images in info.items():
                tblogger.image_summary(tag, images, it)

        viz_time.update(time.time() - start)

        # Run one step of the optimization (controls are always zeros, poses
        # change)
        start = time.time()
        ctrl_grad, loss = optimize_ctrl(model=model.transitionmodel,
                                        poses=curr_poses, ctrl=init_ctrl_v,
                                        goal_poses=goal_poses_v)
        optim_time.update(time.time() - start)
        ctrl_grads[it] = ctrl_grad.cpu().float()  # Save this

        # Set last 3 joint's controls to zero
        if pargs.only_top4_jts:
            ctrl_grad[4:] = 0

        # Get the control direction and scale it by max control magnitude
        start = time.time()
        if ctrl_mag > 0:
            ctrl_dirn = ctrl_grad.cpu().float() / ctrl_grad.norm(2)  # Dirn
            curr_ctrl = ctrl_dirn * ctrl_mag  # Scale dirn by mag
            ctrl_mag *= pargs.ctrl_mag_decay  # Decay control magnitude
        else:
            curr_ctrl = ctrl_grad.cpu().float()

        # v = v - eta*gv
        # Apply control (simple velocity integration)
        # curr_ctrl is not a velocity, it's a change in velocity
        # j = j + v*dt
        # next_angles = curr_angles + (0 - curr_ctrl)* dt
        next_angles = curr_angles - (curr_ctrl * dt)

        # Save stuff
        losses[it] = loss
        ctrls[it] = curr_ctrl
        angles[it + 1] = next_angles
        deg_errors[it + 1] = (next_angles - goal_angles) * (180.0 / np.pi)

        # Print losses and errors
        print('Control Iter: {}/{}, Loss: {}'.format(it + 1, pargs.max_iter, loss))
        print('Joint angle errors in degrees: ',
              torch.cat([deg_errors[it + 1].unsqueeze(1), full_deg_error.unsqueeze(1)], 1))

        # Plot the errors & loss
        if (it % 4) == 0:
            axes[0].set_title("Iter: {}, Jt angle errors".format(it + 1))
            axes[0].plot(deg_errors.numpy()[:it + 1])
            axes[1].set_title("Iter: {}, Loss".format(it + 1))
            axes[1].plot(losses.numpy()[:it + 1])
            fig.canvas.draw()  # Render
            plt.pause(0.01)
        if (it % args.disp_freq) == 0:  # Clear now and then
            for ax in axes:
                ax.cla()

        # Finish
        rest_time.update(time.time() - start)
        print('Gen: {:.3f}({:.3f}), PoseMask: {:.3f}({:.3f}), Viz: {:.3f}({:.3f}),'
              ' Optim: {:.3f}({:.3f}), Rest: {:.3f}({:.3f})'.format(
                  gen_time.val, gen_time.avg, posemask_time.val, posemask_time.avg,
                  viz_time.val, viz_time.avg, optim_time.val, optim_time.avg,
                  rest_time.val, rest_time.avg))

    # Print final stats
    print('=========== FINISHED ============')
    print('Final loss after {} iterations: {}'.format(
        pargs.max_iter, losses[-1]))
    print('Final angle errors in degrees: ')
    print(deg_errors[-1].view(7, 1))

    # Save stats and exit
    stats = {'args': args, 'pargs': pargs, 'data_path': data_path, 'start_id': start_id,
             'goal_id': goal_id, 'start_angles': start_angles, 'goal_angles': goal_angles,
             'angles': angles, 'ctrls': ctrls, 'predctrls': ctrl_grads, 'deg_errors': deg_errors,
             'losses': losses}
    torch.save(stats, pargs.save_dir + '/planstats.pth.tar')

    # TODO: Save errors to file for easy reading??

# Function to generate the optimized control
# Note: assumes that it get Variables


def optimize_ctrl(model, poses, ctrl, goal_poses):

    # Do specific optimization based on the type
    if pargs.optimization == 'backprop':
        # Model has to be in training mode
        model.train()

        # ============ FWD pass + Compute loss ============#

        # FWD pass + loss
        poses_1 = util.to_var(poses.data, requires_grad=False)
        ctrl_1 = util.to_var(ctrl.data, requires_grad=True)
        _, pred_poses = model([poses_1, ctrl_1])
        # Get distance from goal
        loss = args.loss_scale * ctrlnets.BiMSELoss(pred_poses, goal_poses)

        # ============ BWD pass ============#

        # Backward pass & optimize
        model.zero_grad()  # Zero gradients
        zero_gradients(ctrl_1)  # Zero gradients for controls
        loss.backward()  # Compute gradients - BWD pass

        # Return
        return ctrl_1.grad.data.cpu().view(-1, 1).clone(), loss.data[0]
    else:
        # No backprops here
        model.eval()

        # ============ Compute finite differenced controls ============#

        # Setup stuff for perturbation
        eps = pargs.gn_perturb
        nperturb = args.num_ctrl
        I = torch.eye(nperturb).type_as(ctrl.data)

        # Do perturbation
        poses_p = util.to_var(poses.data.repeat(
            nperturb + 1, 1, 1, 1))              # Replicate poses
        # Replicate controls
        ctrl_p = util.to_var(ctrl.data.repeat(nperturb + 1, 1))
        ctrl_p.data[1:, :] += I * eps    # Perturb the controls

        # ============ FWD pass ============#

        # FWD pass
        _, pred_poses_p = model([poses_p, ctrl_p])

        # Backprop only over the loss!
        pred_poses = util.to_var(pred_poses_p.data.narrow(
            0, 0, 1), requires_grad=True)  # Need grad of loss w.r.t true pred
        loss = args.loss_scale * ctrlnets.BiMSELoss(pred_poses, goal_poses)
        loss.backward()

        # ============ Compute Jacobian & GN-gradient ============#

        # Compute Jacobian
        Jt = pred_poses_p.data[1:].view(
            nperturb, -1).clone()  # nperturb x posedim
        # [ f(x+eps) - f(x) ]
        Jt -= pred_poses_p.data.narrow(0, 0, 1).view(1, -1).expand_as(Jt)
        Jt.div_(eps)  # [ f(x+eps) - f(x) ] / eps

        # Option 1: Compute GN-gradient using torch stuff by adding eps * I
        # This is incredibly slow at the first iteration
        # (J^t * J + \lambda I)^-1
        Jinv = torch.inverse(torch.mm(Jt, Jt.t()) + pargs.gn_lambda * I)
        # (J^t*J + \lambda I)^-1 * (Jt * g)
        ctrl_grad = torch.mm(Jinv, torch.mm(
            Jt, pred_poses.grad.data.view(-1, 1)))

        '''
        ### Option 2: Compute GN-gradient using numpy PINV (instead of adding eps * I)
        # Fastest, but doesn't do well on overall planning if we allow controlling all joints
        # If only controlling the top 4 jts this works just as well as the one above.
        Jtn = util.to_np(Jt)
        ctrl_gradn = np.dot(np.linalg.pinv(Jtn, rcond=pargs.gn_lambda).transpose(), util.to_np(pred_poses.grad.data.view(-1,1)))
        ctrl_grad  = torch.from_numpy(ctrl_gradn)
        '''

        '''
        ### Option 3: Compute GN-gradient using numpy INV (add eps * I)
        # Slower than torch
        Jtn, In = util.to_np(Jt), util.to_np(I)
        Jinv = np.linalg.inv(np.dot(Jtn, Jtn.transpose()) + pargs.gn_lambda * In) # (J^t * J + \lambda I)^-1
        ctrl_gradn = np.dot(Jinv, np.dot(Jtn, util.to_np(pred_poses.grad.data.view(-1,1))))
        ctrl_grad  = torch.from_numpy(ctrl_gradn)
        '''

        # ============ Sanity Check stuff ============#
        # Check gradient / jacobian
        if pargs.gn_jac_check:
            # Set model in training mode
            model.train()

            # FWD pass
            poses_1 = util.to_var(poses.data, requires_grad=False)
            ctrl_1 = util.to_var(ctrl.data, requires_grad=True)
            _, pred_poses_1 = model([poses_1, ctrl_1])
            pred_poses_1_v = pred_poses_1.view(1, -1)  # View it nicely

            ###
            # Compute Jacobian via multiple backward passes (SLOW!)
            Jt_1 = compute_jacobian(ctrl_1, pred_poses_1_v)
            diff_j = Jt.t() - Jt_1
            print('Jac diff => Min: {}, Max: {}, Mean: {}'.format(
                diff_j.min(), diff_j.max(), diff_j.abs().mean()))

            ###
            # Compute gradient via single backward pass + loss
            # Get distance from goal
            loss = args.loss_scale * \
                ctrlnets.BiMSELoss(pred_poses_1, goal_poses)
            model.zero_grad()  # Zero gradients
            zero_gradients(ctrl_1)  # Zero gradients for controls
            loss.backward()  # Compute gradients - BWD pass
            # Error between backprop & J^T g from FD
            diff_g = ctrl_1.grad.data - \
                torch.mm(Jt, pred_poses.grad.data.view(-1, 1))
            print('Grad diff => Min: {}, Max: {}, Mean: {}'.format(
                diff_g.min(), diff_g.max(), diff_g.abs().mean()))

        # Return the Gauss-Newton gradient
        return ctrl_grad.cpu().view(-1).clone(), loss.data[0]

# Compute a point cloud give the arm config
# Assumes that a "Tensor" is the input, not a "Variable"


def generate_ptcloud(config):
    # Render the config & get the point cloud
    assert(not util.is_var(config))
    config_f = config.view(-1).clone().float()
    pts = torch.FloatTensor(1, 3, args.img_ht, args.img_wd)
    pangolin.render_arm(config_f.numpy(), pts[0].numpy())
    return pts.type_as(config)

# Compute numerical jacobian via multiple back-props


def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    num_outputs = output.size()[1]

    jacobian = torch.zeros(num_outputs, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_outputs):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_variables=True)
        jacobian[i] = inputs.grad.data

    return jacobian


# RUN MAIN
if __name__ == '__main__':
    main()
