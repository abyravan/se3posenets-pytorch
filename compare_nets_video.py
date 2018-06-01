# Global imports
import _init_paths

# TODO: Make this cleaner, we don't need most of these parameters to create the pangolin window
img_ht, img_wd, img_scale = 240, 320, 1e-4
cam_intrinsics = {'fx': 589.3664541825391 / 2,
                  'fy': 589.3664541825391 / 2,
                  'cx': 320.5 / 2,
                  'cy': 240.5 / 2}

# Load pangolin visualizer library
try:
    from torchviz import realctrlviz # Had to add this to get the file to work, otherwise gave static TLS error
    from torchviz import compviz
    pangolin = compviz.PyCompViz(img_ht, img_wd, img_scale,
                                 cam_intrinsics['fx'], cam_intrinsics['fy'],
                                 cam_intrinsics['cx'], cam_intrinsics['cy'], 1)
except:
    print('Running without Pangolin')
    pass

# Global imports
import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import se3nets
import flownets
import util
from util import AverageMeter, Tee, DataEnumerator

#### Setup options
# Common
import argparse

# Display/Save options
parser = argparse.ArgumentParser(description='Compare networks')
parser.add_argument('-s', '--save-dir', default='temp', type=str,
                    metavar='PATH', help='directory to save results in. (default: temp)')
parser.add_argument('--save-frames', action='store_true', default=False,
                    help='Enables post-saving of generated frames, very slow process (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--num-examples', type=int, default=100, metavar='N',
                    help='Num test examples to run on (default: 100)')
parser.add_argument('--seq-len', type=int, default=10, metavar='N',
                    help='Length of the sampled sequence (default: 10)')
parser.add_argument('--real-data', action='store_true', default=False,
                    help='test on real data (default: False)')
parser.add_argument('--large-motions-only', action='store_true', default=False,
                    help='test only on examples which have large motions (default: False)')

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ MAIN
#@profile
def main():

    ########################
    ############ Parse options
    # Parse args
    global cargs, num_train_iter
    cargs = parser.parse_args()
    cargs.cuda = not cargs.no_cuda and torch.cuda.is_available()

    ### Create save directory and start tensorboard logger
    util.create_dir(cargs.save_dir)  # Create directory

    # # Create logfile to save prints
    # logfile = open(cargs.save_dir + '/logfile.txt', 'w')
    # backup = sys.stdout
    # sys.stdout = Tee(sys.stdout, logfile)

    # Set seed
    torch.manual_seed(cargs.seed)
    if cargs.cuda:
        torch.cuda.manual_seed(cargs.seed)

    # # If we are asked to save frames:
    # if cargs.save_frames:
    #     save_dir = cargs.save_dir + "/frames/"
    #     util.create_dir(save_dir)  # Create directory
    #     pangolin.start_saving_frames(save_dir)  # Start saving frames

    ########################################
    ############ Load trained networks!
    models = {}
    num_train_iter = {}

    ########
    ## TODO: Load SE3-Pose-Net
    if cargs.real_data:
        #posemodelfile = "icra18results/networks/realdata/oursbestfixed/8se3_wtsharpenr1s0_1seq_normmsesqrt_motionnormloss_rlcheck/model_best.pth.tar"
        posemodelfile = "icra18results/finalsubmission/networks/realdata/se3pose/def_rmsprop/model_flow_best.pth.tar"
    else:
        #posemodelfile = "icra18results/networks/alldata/oursbestfixed/8se3_wtsharpenr1s0_1seq_normmsesqrt_motionnormloss/model_best.pth.tar"
        posemodelfile = "icra18results/finalsubmission/networks/simdata/se3pose/def_rmsprop/model_flow_best.pth.tar"
    print("=> loading SE3-Pose-Net")
    checkpoint_pm = torch.load(posemodelfile)
    print("=> loaded checkpoint (epoch: {}, num train iter: {})"
              .format(checkpoint_pm['epoch'], checkpoint_pm['train_iter']))

    #### Common args
    args = checkpoint_pm['args'] # Use as default args
    # BWDs compatibility
    if not hasattr(args, "use_gt_masks"):
        args.use_gt_masks, args.use_gt_poses = False, False
    if not hasattr(args, "num_state"):
        args.num_state = args.num_ctrl
    if not hasattr(args, "use_gt_angles"):
        args.use_gt_angles, args.use_gt_angles_trans = False, False
    if not hasattr(args, "num_state"):
        args.num_state = 7
    if not hasattr(args, "mean_dt"):
        args.mean_dt = args.step_len * (1.0/30.0)
        args.std_dt  = 0.005 # Default params
    if not hasattr(args, "delta_pivot"):
        args.delta_pivot = ''
    if not hasattr(args, "pose_center"):
        args.pose_center = 'pred'

    if not hasattr(args, "use_full_jt_angles"):
        args.use_full_jt_angles = True
    if args.use_full_jt_angles:
        args.num_state_net = args.num_state
    else:
        args.num_state_net = args.num_ctrl

    ##### Get default options & camera intrinsics
    args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
    args.state_labels = []
    for k in xrange(len(args.data)):
        load_dir = args.data[k]  # args.data.split(',,')[0]
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
            cam_intrinsics = {'fx': intrinsics['fx'] * sc,
                              'fy': intrinsics['fy'] * sc,
                              'cx': intrinsics['cx'] * sc,
                              'cy': intrinsics['cy'] * sc}
            print("Scale factor for the intrinsics: {}".format(sc))
        except:
            print("Could not read intrinsics file, reverting to default settings")
            args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
            cam_intrinsics = {'fx': 589.3664541825391 / 2,
                              'fy': 589.3664541825391 / 2,
                              'cx': 320.5 / 2,
                              'cy': 240.5 / 2}
        print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(args.img_ht, args.img_wd,
                                                                                    cam_intrinsics['fx'],
                                                                                    cam_intrinsics['fy'],
                                                                                    cam_intrinsics['cx'],
                                                                                    cam_intrinsics['cy']))

        # Compute intrinsic grid
        cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                              cam_intrinsics)

        # Compute intrinsics
        cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

        # Get dimensions of ctrl & state
        try:
            statelabels, ctrllabels, trackerlabels = data.read_statectrllabels_file(
                load_dir + "/statectrllabels.txt")
            print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
        except:
            statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
            ctrllabels = statelabels  # Just use the labels
            trackerlabels = []
            print("Could not read statectrllabels file. Reverting to labels in statelabels file")
        # args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
        # print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
        args.num_ctrl = len(ctrllabels)
        print('Num ctrl: {}'.format(args.num_ctrl))

        # Find the IDs of the controlled joints in the state vector
        # We need this if we have state dimension > ctrl dimension and
        # if we need to choose the vals in the state vector for the control
        ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
        print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

        # Add to list of intrinsics
        args.cam_intrinsics.append(cam_intrinsics)
        args.cam_extrinsics.append(cam_extrinsics)
        args.ctrl_ids.append(ctrlids_in_state)
        args.state_labels.append(statelabels)

    # Data noise
    if not hasattr(args, "add_noise_data") or (len(args.add_noise_data) == 0):
        args.add_noise_data = [False for k in xrange(len(args.data))]  # By default, no noise
    else:
        assert (len(args.data) == len(args.add_noise_data))
    if hasattr(args, "add_noise") and args.add_noise:  # BWDs compatibility
        args.add_noise_data = [True for k in xrange(len(args.data))]

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

    ########
    ## Multi-step model
    num_train_iter['se3pose'] = checkpoint_pm['train_iter']
    models['se3pose'] = ctrlnets.MultiStepSE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                           se3_type=args.se3_type, delta_pivot=args.delta_pivot, use_kinchain=False,
                                           input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                           init_posese3_iden=args.init_posese3_iden,
                                           init_transse3_iden=args.init_transse3_iden,
                                           use_wt_sharpening=args.use_wt_sharpening,
                                           sharpen_start_iter=args.sharpen_start_iter,
                                           sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                                           decomp_model=args.decomp_model,
                                           use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                                           wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                                           use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state_net)
    if args.cuda:
        models['se3pose'].cuda()  # Convert to CUDA if enabled
    models['se3pose'].load_state_dict(checkpoint_pm['model_state_dict'])
    models['se3pose'].eval() # Set to eval mode

    ########
    ## TODO: Load SE3-Net
    if cargs.real_data:
        #se3modelfile = "icra18results/networks/realdata/se3/8se3_wtsharpenr1s0_1seq_normmsesqrt_motionnormloss_rlcheck/model_best.pth.tar"
        se3modelfile = "icra18results/finalsubmission/networks/realdata/se3/def_rate0.25_1/model_best.pth.tar"
    else:
        #se3modelfile = "icra18results/networks/alldata/se3/8se3_wtsharpenr1s0_1seq_normmsesqrt_motionnormloss/model_best.pth.tar"
        se3modelfile = "icra18results/finalsubmission/networks/simdata/se3/def_1/model_best.pth.tar"
    print("=> loading SE3-Net")
    checkpoint_sm = torch.load(se3modelfile)
    print("=> loaded checkpoint (epoch: {}, num train iter: {})"
          .format(checkpoint_sm['epoch'], checkpoint_sm['train_iter']))

    ## Multi-step model
    num_train_iter['se3'] = checkpoint_sm['train_iter']
    models['se3'] = se3nets.SE3Model(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                 se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                                 input_channels=3, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                                 init_transse3_iden=args.init_transse3_iden,
                                 use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                                 sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                                 wide=args.wide_model, se2_data=False, use_jt_angles=args.use_jt_angles,
                                 num_state=args.num_state_net, use_lstm=(args.seq_len > 1))
    if args.cuda:
        models['se3'].cuda()  # Convert to CUDA if enabled
    models['se3'].load_state_dict(checkpoint_sm['model_state_dict'])
    models['se3'].eval()  # Set to eval mode

    ########
    ## TODO: Load Flow-Net
    if cargs.real_data:
        #flowmodelfile = "icra18results/networks/realdata/flow/1seq_normmsesqrt_motionnormloss_rlcheck/model_best.pth.tar"
        flowmodelfile = "icra18results/finalsubmission/networks/realdata/flow/def/model_best.pth.tar"
    else:
        #flowmodelfile = "icra18results/networks/alldata/flow/1seq_normmsesqrt_motionnormloss/model_best.pth.tar"
        flowmodelfile = "icra18results/finalsubmission/networks/simdata/flow/def/model_best.pth.tar"
    print("=> loading Flow-Network")
    checkpoint_fm = torch.load(flowmodelfile)
    print("=> loaded checkpoint (epoch: {}, num train iter: {})"
          .format(checkpoint_fm['epoch'], checkpoint_fm['train_iter']))

    ## Multi-step model
    num_train_iter['flow'] = checkpoint_fm['train_iter']
    models['flow'] = flownets.FlowNet(num_ctrl=args.num_ctrl, num_state=args.num_state_net,
                                 input_channels=3, use_bn=args.batch_norm, pre_conv=args.pre_conv,
                                 nonlinearity=args.nonlin, init_flow_iden=False,
                                 use_jt_angles=args.use_jt_angles, use_lstm=(args.seq_len > 1))
    if args.cuda:
        models['flow'].cuda()  # Convert to CUDA if enabled
    models['flow'].load_state_dict(checkpoint_fm['model_state_dict'])
    models['flow'].eval()  # Set to eval mode

    ########################################
    ############ Load datasets
    if not cargs.real_data:
        args.data = ['/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions_f']
    # Get datasets
    if args.reject_left_motion:
        print("Examples where any joint of the left arm moves by > 0.005 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.reject_right_still:
        print("Examples where no joint of the right arm move by > 0.015 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.add_noise:
        print("Adding noise to the depths, actual configs & ctrls")
    #noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
    #                                                  scale_d=True, std_j=0.02) if args.add_noise else None
    noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                     defprob=0.005, noisestd=0.005)
    valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                               mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                               reject_left_motion=args.reject_left_motion,
                                                               reject_right_still=args.reject_right_still)
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = cargs.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per,
                                                         valid_filter = valid_filter,
                                                         cam_extrinsics=args.cam_extrinsics,
                                                         cam_intrinsics=args.cam_intrinsics,
                                                         ctrl_ids=args.ctrl_ids,
                                                         state_labels=args.state_labels,
                                                         add_noise=args.add_noise_data)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                       img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                                       num_ctrl=args.num_ctrl,
                                                                       #num_state=args.num_state,
                                                                       mesh_ids = args.mesh_ids,
                                                                       #ctrl_ids=ctrlids_in_state,
                                                                       #camera_extrinsics = args.cam_extrinsics,
                                                                       #camera_intrinsics = args.cam_intrinsics,
                                                                       compute_bwdflows=True,
                                                                       #num_tracker=args.num_tracker,
                                                                       dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                                       use_only_da=args.use_only_da_for_flows) # Need BWD flows / masks if using GT masks
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Test dataset size => {}'.format(len(test_dataset)))

    # test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=1, shuffle=True,
    #                                              num_workers=args.num_workers, pin_memory=args.use_pin_memory,
    #                                              collate_fn=test_dataset.collate_batch))

    ########################
    ############ Eval through the models & predict/render
    data_time, fwd_time, viz_time = AverageMeter(), AverageMeter(), AverageMeter()
    stats = argparse.Namespace()
    stats.flowerr_sum, stats.flowerr_avg = {'se3pose': AverageMeter(), 'se3posepts': AverageMeter(), 'se3': AverageMeter(), 'flow': AverageMeter()}, \
                                           {'se3pose': AverageMeter(), 'se3posepts': AverageMeter(), 'se3': AverageMeter(), 'flow': AverageMeter()}
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'  # Default tensor type

    example_ids = [3572, 33253, 38826, 27607, 65465, 26246, 33928, 22026, 69030, 88021, 8318, 44744,
                   63415, 42957, 54401, 49726, 61120, 29602, 84379, 59432, 86434]

    sumerrs = torch.zeros(4, cargs.seq_len, cargs.num_examples)
    avgerrs = torch.zeros(4, cargs.seq_len, cargs.num_examples)
    ids = []
    ctr = 0
    #for id in example_ids:
    while len(ids) < cargs.num_examples:
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # If we are asked to save frames:
        ctr = len(ids)
        if cargs.save_frames:
            save_dir = cargs.save_dir + "/frames/test{}".format(ctr)
            print("Saving frames in folder: {}".format(save_dir))
            util.create_dir(save_dir)  # Create directory
            pangolin.start_saving_frames(save_dir)  # Start saving frames
            #ctr+=1

            # Create logfile to save prints
            logfile = open(save_dir + '/logfile.txt', 'w')
            backup = sys.stdout
            sys.stdout = Tee(sys.stdout, logfile)

        # Get a sample & pre-check if there is good motion in the sample
        id = random.randint(0,len(test_dataset)-100)

        ## Threshold for large motions (discard examples with little motion)
        if cargs.large_motions_only:
            # Load entire sequence of configs
            sequence, path = data.generate_baxter_sequence(test_dataset.datasets[0], id)  # Get the file paths
            actctrlconfigs = torch.FloatTensor(cargs.seq_len + 1, len(args.ctrl_ids[0]))  # Ids in actual data belonging to commanded data
            for k in xrange(len(sequence)):
                state = data.read_baxter_state_file(sequence[k]['state1'])
                actctrlconfigs[k] = state['actjtpos'][args.ctrl_ids[0]]  # Get states for control IDs
            nstill = (((actctrlconfigs[:-1] - actctrlconfigs[1:]).abs_().max(1))[0] < 0.05).sum()
            rightok = (nstill < cargs.seq_len / 3)  # Atleast half the frames need to have motion
            if not rightok:
                print("Discarded example as num still frames: {} > {}".format(nstill, cargs.seq_len/3))
                continue # Sample again

        # Append to list of IDs
        ids.append(id)

        # Get full dataset
        sample = test_dataset[id]

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts      = util.to_var(sample['points'].unsqueeze(0).type(deftype), requires_grad=False, volatile=True)
        ctrls    = util.to_var(sample['controls'].unsqueeze(0).type(deftype), requires_grad=False)
        tarpts   = util.to_var(sample['fwdflows'].unsqueeze(0).type(deftype), requires_grad=False)
        tarpts.data.add_(pts.data.narrow(1,0,1).expand_as(tarpts.data))  # Add "k"-step flows to the initial point cloud

        # Get jt angles
        jtangles = util.to_var(sample['actctrlconfigs'].unsqueeze(0).type(deftype), requires_grad=False)  # [:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        #################################################################################
        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()
        predpts = {}

        ########
        ### TODO: FWD pass through SE3-Pose-Net
        ### Run a FWD pass through the network (multi-step)
        # Predict the initpose and initmask
        initpose, initmask = models['se3pose'].forward_pose_mask([pts[:,0], jtangles[:,0]], train_iter=num_train_iter['se3pose'])

        # Run sequence of controls through transition model only to predict sequence of deltas!
        # We use pose_0 and [ctrl_0, ctrl_1, .., ctrl_(T-1)] to make the predictions
        deltaposes, transposes, compdeltaposes = [], [], []
        for k in xrange(cargs.seq_len):
            # Get current pose
            pose = initpose if (k == 0) else transposes[k-1]

            # Predict next pose based on curr pose, control
            delta, trans = models['se3pose'].forward_next_pose(pose, ctrls[:,k], jtangles[:,k])
            deltaposes.append(delta)
            transposes.append(trans)

        # Now compute the sequence of predicted pt clouds
        # We use point loss in the FWD dirn and Consistency loss between poses
        predpts['se3pose'] = []
        for k in xrange(cargs.seq_len):
            # Predict transformed 3D points
            currpts = pts[:,0] if (k == 0) else predpts['se3pose'][k-1]
            nextpts = se3nn.NTfm3D()(currpts, initmask, deltaposes[k])  # We do want to backpropagate to all the deltas in this case
            predpts['se3pose'].append(nextpts)  # Save predicted pts

        ########
        ### TODO: FWD pass through SE3-Pose-Net -> no rollouts in the pose space!
        ### Run a FWD pass through the network (multi-step)
        masks_se3posepts = []
        predpts['se3posepts'] = []
        for k in xrange(cargs.seq_len):
            # Get current pts & make a prediction
            currpts = pts[:,0] if (k == 0) else predpts['se3posepts'][k-1]
            pose, mask = models['se3pose'].forward_pose_mask([currpts, jtangles[:,k]], train_iter=num_train_iter['se3pose'])

            # Predict next pose based on curr pose, control
            delta, trans = models['se3pose'].forward_next_pose(pose, ctrls[:,k], jtangles[:,k])

            # Predict next pts based on current pts
            nextpts = se3nn.NTfm3D()(currpts, mask, delta)  # We do want to backpropagate to all the deltas in this case
            predpts['se3posepts'].append(nextpts)
            masks_se3posepts.append(mask.clone())

        ########
        ### TODO: FWD pass through SE3-Net
        ### Run a FWD pass through the network (multi-step)
        predpts['se3'] = []
        masks_se3, deltaposes_se3 = [], []
        for k in xrange(cargs.seq_len):
            # Get current input to network
            currpts = pts[:,0] if (k == 0) else predpts['se3'][k-1]

            # Make flow prediction
            flows, [deltapose, mask] = models['se3']([currpts, jtangles[:,k], ctrls[:,k]],
                                                      reset_hidden_state=(k==0), train_iter = num_train_iter['se3']) # Reset hidden state at start of sequence
            deltaposes_se3.append(deltapose)
            masks_se3.append(mask.clone())

            # Get next predicted pts
            predpts['se3'].append(currpts + flows)

        ########
        ### TODO: FWD pass through Flow-Network
        ### Run a FWD pass through the network (multi-step)
        predpts['flow'] = []
        for k in xrange(cargs.seq_len):
            # Get current input to network
            currpts = pts[:,0] if (k == 0) else predpts['flow'][k-1]

            # Make flow prediction
            flows = models['flow']([currpts, jtangles[:, k], ctrls[:, k]],
                                    reset_hidden_state=(k == 0))  # Reset hidden state at start of sequence

            # Get next predicted pts
            predpts['flow'].append(currpts + flows)

        # Measure FWD time
        fwd_time.update(time.time() - start)

        #################################################################################
        ### Compute flow errors
        predflows = {}
        flowerr_sum, flowerr_avg = {}, {}
        flows_t     = sample['fwdflows'].unsqueeze(0).type(deftype)
        for ky, _ in predpts.items():
            predflows[ky] = torch.cat([(x.data - pts.data[:, 0]).unsqueeze(1) for x in predpts[ky]], 1)
            flowerr_sum[ky], flowerr_avg[ky], _,_,_,_,\
            _,_,_,_ = compute_masked_flow_errors(predflows[ky], flows_t)

        ### Save errs
        sumerrs[0,:,ctr], sumerrs[1,:,ctr], \
        sumerrs[2,:,ctr], sumerrs[3,:,ctr] = flowerr_sum['se3pose'], flowerr_sum['se3posepts'], \
                                                 flowerr_sum['se3'], flowerr_sum['flow']
        avgerrs[0,:,ctr], avgerrs[1,:,ctr], \
        avgerrs[2,:,ctr], avgerrs[3,:,ctr] = flowerr_avg['se3pose'], flowerr_avg['se3posepts'], \
                                                 flowerr_avg['se3'], flowerr_avg['flow']

        ### Print stuff
        print('\tExample: {}/{}. ID: {}. '
                'SE3-Pose => Sum/Avg: {:.3f}/{:.3f}, '
                'SE3-Pose-Pts => Avg: {:.3f}/{:.3f}, '
                'SE3 => Avg: {:.3f}/{:.3f}, '
                'Flow => Avg: {:.3f}/{:.3f})'.format(len(ids), cargs.num_examples, id, #len(example_ids), id,
            flowerr_avg['se3pose'].sum(), flowerr_avg['se3pose'].mean(),
            flowerr_avg['se3posepts'].sum(), flowerr_avg['se3posepts'].mean(),
            flowerr_avg['se3'].sum(), flowerr_avg['se3'].mean(),
            flowerr_avg['flow'].sum(), flowerr_avg['flow'].mean(),
        ))
        for k in xrange(0,cargs.seq_len,2):
            print('\tStep: {}, '
                  'SE3-Pose => Sum: {:4.3f}, Avg: {:2.3f}, '
                  'SE3-Pose-Pts => Sum: {:4.3f}, Avg: {:2.3f}, '
                  'SE3 => Sum: {:4.3f}, Avg: {:2.3f}, '
                  'Flow => Sum: {:4.3f}, Avg: {:2.3f}'
                .format(
                1 + k * args.step_len,
                flowerr_sum['se3pose'][k],
                flowerr_avg['se3pose'][k],
                flowerr_sum['se3posepts'][k],
                flowerr_avg['se3posepts'][k],
                flowerr_sum['se3'][k],
                flowerr_avg['se3'][k],
                flowerr_sum['flow'][k],
                flowerr_avg['flow'][k],
            ))

        ## Display using pangolin
        for j in xrange(len(predpts['se3pose'])):
            if (j % 10 == 0):
                print("Displaying frame: {}/{}".format(j, len(predpts['se3pose'])))
            pangolin.update(pts[0,0].data.cpu().numpy(),
                            #pts[0,j+1].data.cpu().numpy(),
                            tarpts[0,j].data.cpu().numpy(),
                            predpts['se3pose'][j][0].data.cpu().numpy(),
                            predpts['se3posepts'][j][0].data.cpu().numpy(),
                            predpts['se3'][j][0].data.cpu().numpy(),
                            predpts['flow'][j][0].data.cpu().numpy(),
                            #sample['masks'].unsqueeze(0)[0,j+1].float().numpy(),
                            sample['masks'].unsqueeze(0)[0,0].float().numpy(),
                            initmask[0].data.cpu().numpy(),
                            masks_se3posepts[j][0].data.cpu().numpy(),
                            masks_se3[j][0].data.cpu().numpy(),
                            cargs.save_frames)  # Save frame
            time.sleep(0.1)

        # Stop saving frames
        if cargs.save_frames:
            time.sleep(1)
            pangolin.stop_saving_frames()
            sys.stdout = backup
            torch.save({"id":id, "err_sum": flowerr_sum,
                        "err_avg": flowerr_avg}, save_dir+"/data.pth.tar")

        predpts, sample, predflows, masks_se3, masks_se3posepts, initmask, pts, tarpts, flows_t = [None]*9
        import gc; gc.collect()

    ## Final result
    meanavg, stdavg = avgerrs.mean(2), avgerrs.std(2)
    print("MEAN")
    print(meanavg.t())
    print("STD")
    print(stdavg.t())
    torch.save({"cargs": cargs, "ids": ids, "sumerrs": sumerrs, "avgerrs": avgerrs},
               cargs.save_dir+"/data.pth.tar")

### Compute flow errors for moving / non-moving pts (flows are size: B x S x 3 x H x W)
def compute_masked_flow_errors(predflows, gtflows):
    batch, seq = predflows.size(0), predflows.size(1) # B x S x 3 x H x W
    # Compute num pts not moving per mask
    # !!!!!!!!! > 1e-3 returns a ByteTensor and if u sum within byte tensors, the max value we can get is 255 !!!!!!!!!
    motionmask = (gtflows.abs().sum(2) > 1e-3).type_as(gtflows) # B x S x 1 x H x W
    err = (predflows - gtflows).mul_(1e2).pow(2).sum(2) # B x S x 1 x H x W

    # Compute errors for points that are supposed to move
    motion_err = (err * motionmask).view(batch, seq, -1).sum(2) # Errors for only those points that are supposed to move
    motion_npt = motionmask.view(batch, seq, -1).sum(2) # Num points that move (B x S)

    # Compute errors for points that are supposed to not move
    motionmask.eq_(0) # Mask out points that are not supposed to move
    still_err = (err * motionmask).view(batch, seq, -1).sum(2)  # Errors for non-moving points
    still_npt = motionmask.view(batch, seq, -1).sum(2)  # Num non-moving pts (B x S)

    # Bwds compatibility to old error
    full_err_avg  = (motion_err + still_err) / motion_npt
    full_err_avg[full_err_avg != full_err_avg] = 0  # Clear out any Nans
    full_err_avg[full_err_avg == np.inf] = 0  # Clear out any Infs
    full_err_sum, full_err_avg = (motion_err + still_err).sum(0), full_err_avg.sum(0) # S, S

    # Compute sum/avg stats
    motion_err_avg = (motion_err / motion_npt)
    motion_err_avg[motion_err_avg != motion_err_avg] = 0  # Clear out any Nans
    motion_err_avg[motion_err_avg == np.inf] = 0      # Clear out any Infs
    motion_err_sum, motion_err_avg = motion_err.sum(0), motion_err_avg.sum(0) # S, S

    # Compute sum/avg stats
    still_err_avg = (still_err / still_npt)
    still_err_avg[still_err_avg != still_err_avg] = 0  # Clear out any Nans
    still_err_avg[still_err_avg == np.inf] = 0  # Clear out any Infs
    still_err_sum, still_err_avg = still_err.sum(0), still_err_avg.sum(0)  # S, S

    # Return
    return full_err_sum.cpu().float(), full_err_avg.cpu().float(), \
           motion_err_sum.cpu().float(), motion_err_avg.cpu().float(), \
           still_err_sum.cpu().float(), still_err_avg.cpu().float(), \
           motion_err.cpu().float(), motion_npt.cpu().float(), \
           still_err.cpu().float(), still_npt.cpu().float()

################ RUN MAIN
if __name__ == '__main__':
    main()
