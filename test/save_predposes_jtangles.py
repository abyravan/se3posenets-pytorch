# Global imports
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import time

# Torch imports
import torch
import torch.optim
import torch.utils.data
torch.multiprocessing.set_sharing_strategy('file_system')

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

# Loss options
parser.add_argument('--backprop-only-first-delta', action='store_true', default=False,
                    help='Backprop gradients only to the first delta. Switches from using delta-flow-loss to'
                         'full-flow-loss with copied composed deltas if this is set (default: False)')
parser.add_argument('--pt-wt', default=1, type=float,
                    metavar='WT', help='Weight for the 3D point loss - only FWD direction (default: 1)')
parser.add_argument('--use-full-jt-angles', action='store_true', default=False,
                    help='Use angles of all joints as inputs to the networks (default: False)')

# Pivot options
parser.add_argument('--pose-center', default='pred', type=str,
                    metavar='STR', help='Different options for pose center positions: [pred] | predwmaskmean | predwmaskmeannograd')
parser.add_argument('--delta-pivot', default='', type=str,
                    metavar='STR', help='Pivot prediction for the delta-tfm: [] | pred | ptmean | maskmean | '
                                        'maskmeannograd | posecenter')
parser.add_argument('--consis-rt-loss', action='store_true', default=False,
                    help='Use RT loss for the consistency measure (default: False)')

# Loss between pose center and mask mean
parser.add_argument('--pose-anchor-wt', default=0.0, type=float,
                    metavar='WT', help='Weight for the loss anchoring pose center to be close to the mean mask value')
parser.add_argument('--pose-anchor-maskbprop', action='store_true', default=False,
                    help='Backprop gradient from pose anchor loss to mask mean if set to true (default: False)')
parser.add_argument('--pose-anchor-grad-clip', default=0.0, type=float,
                    metavar='WT', help='Gradient clipping for the pose anchor gradient (to account for outliers) (default: 0.0)')

# Box data
parser.add_argument('--box-data', action='store_true', default=False,
                    help='Dataset has box/ball data (default: False)')

# Use normal data
parser.add_argument('--normal-wt', default=0.0, type=float,
                    metavar='WT', help='Weight for the cosine distance of normal loss (default: 1)')
parser.add_argument('--normal-max-depth-diff', default=0.05, type=float,
                    metavar='WT', help='Max allowed depth difference for a valid normal computation (default: 0.05)')
parser.add_argument('--motion-norm-normal-loss', action='store_true', default=False,
                    help='normalize the normal loss by number of points that actually move instead of all pts (default: False)')
parser.add_argument('--bilateral-depth-smoothing', action='store_true', default=False,
                    help='do bilateral depth smoothing before computing normals (default: False)')
parser.add_argument('--bilateral-window-width', default=9, type=int,
                    metavar='K', help='Size of window for bilateral filtering (default: 9x9)')
parser.add_argument('--bilateral-depth-std', default=0.005, type=float,
                    metavar='WT', help='Standard deviation in depth for bilateral filtering kernel (default: 0.005)')

# Supervised segmentation loss
parser.add_argument('--seg-wt', default=0.0, type=float,
                    metavar='WT', help='Weight for a supervised mask segmentation loss (both @ t & t+1)')

# Transition model type
parser.add_argument('--trans-type', default='default', type=str,
                    metavar='TRANS', help='type of transition model: [default] | linear | simple | simplewide | '
                                          'locallinear | locallineardelta')

# Pose-Mask model type
parser.add_argument('--posemask-type', default='default', type=str,
                    metavar='POSEMASK', help='type of pose-mask model: [default] | unet')
parser.add_argument('--save-filename', default='predposedata-sim.pth.tar', type=str,
                    metavar='PATH', help='file name to save results in (default: gtposedata-sim.pth.tar)')

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ Data loading code
### Load baxter sequence from disk
def read_baxter_sequence_from_disk(dataset, id, img_ht=240, img_wd=320, img_scale=1e-4,
                                   ctrl_type='actdiffvel', num_ctrl=7,
                                   mesh_ids=torch.Tensor(),
                                   compute_bwdflows=True, load_color=None, num_tracker=0,
                                   dathreshold=0.01, dawinsize=5, use_only_da=False,
                                   noise_func=None, compute_normals=False, maxdepthdiff=0.05,
                                   bismooth_depths=False, bismooth_width=9, bismooth_std=0.001,
                                   compute_bwdnormals=False, supervised_seg_loss=False):
    # Setup vars
    num_meshes = mesh_ids.nelement()  # Num meshes
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics, camera_extrinsics, ctrl_ids = dataset['camintrinsics'], dataset['camextrinsics'], dataset['ctrlids']

    # Setup memory
    sequence, path, folid = data.generate_baxter_sequence(dataset, id)  # Get the file paths
    points         = torch.FloatTensor(seq_len, 3, img_ht, img_wd)
    actctrlconfigs = torch.FloatTensor(seq_len, num_ctrl)  # Ids in actual data belonging to commanded data
    poses          = torch.FloatTensor(seq_len, mesh_ids.nelement() + 1, 3, 4).zero_()
    allposes       = torch.FloatTensor()

    # Setup temp var for depth
    depths = points.narrow(1, 2, 1)  # Last channel in points is the depth

    # Setup vars for color image
    if load_color:
        rgbs = torch.ByteTensor(seq_len, 3, img_ht, img_wd)

    ## Read camera extrinsics (can be separate per dataset now!)
    try:
        camera_extrinsics = data.read_cameradata_file(path + '/cameradata.txt')
    except:
        pass  # Can use default cam extrinsics for the entire dataset

    #####
    # Load sequence
    for k in xrange(len(sequence)-1): ## Only do this for the first element!
        # Get data table
        s = sequence[k]

        # Load depth
        depths[k] = data.read_depth_image(s['depth'], img_ht, img_wd, img_scale)  # Third channel is depth (x,y,z)

        # Load configs
        state = data.read_baxter_state_file(s['state1'])
        actctrlconfigs[k] = state['actjtpos'][ctrl_ids]  # Get states for control IDs

        # Load RGB
        if load_color:
            rgbs[k] = data.read_color_image(s['color'], img_ht, img_wd, colormap=load_color)
            # actctrlvels[k] = state['actjtvel'][ctrl_ids] # Get vels for control IDs
            # comvels[k] = state['comjtvel']

        # Load SE3 state & get all poses
        se3state = data.read_baxter_se3state_file(s['se3state1'])
        if allposes.nelement() == 0:
            allposes.resize_(seq_len, len(se3state) + 1, 3, 4).fill_(0)  # Setup size
        allposes[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        for id, tfm in se3state.items():
            se3tfm = torch.mm(camera_extrinsics['modelView'],
                              tfm)  # NOTE: Do matrix multiply, not * (cmul) here. Camera data is part of options
            allposes[k][id] = se3tfm[0:3, :]  # 3 x 4 transform (id is 1-indexed already, 0 is BG)

        # Get poses of meshes we are moving
        poses[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        for j in xrange(num_meshes):
            meshid = mesh_ids[j]
            poses[k][j + 1] = allposes[k][meshid][0:3, :]  # 3 x 4 transform

    # Compute x & y values for the 3D points (= xygrid * depths)
    xy = points[:, 0:2]
    xy.copy_(camera_intrinsics['xygrid'].expand_as(xy))  # = xygrid
    xy.mul_(depths.expand(seq_len, 2, img_ht, img_wd))  # = xygrid * depths

    # Return loaded data
    dataout = {'points': points, 'folderid': folid,
               'poses': poses, 'actctrlconfigs': actctrlconfigs}
    if load_color:
        dataout['rgbs'] = rgbs
    return dataout


################ MAIN
#@profile
def main():
    # Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()
    args.batch_norm = not args.no_batch_norm

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 480 x 640 or 240 x 320
    if args.full_res:
        print("Using full-resolution images (480x640)")
    # XYZ-RGB
    if args.use_xyzrgb:
        print("Using XYZ-RGB input - 6 channels. Assumes registered depth/RGB")
    elif args.use_xyzhue:
        print("Using XYZ-Hue input - 4 channels. Assumes registered depth/RGB")

    # Get default options & camera intrinsics
    args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
    args.state_labels = []
    for k in xrange(len(args.data)):
        load_dir = args.data[k] #args.data.split(',,')[0]
        try:
            # Read from file
            intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
            print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
            if args.se2_data or args.full_res:
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

        # Compute intrinsic grid & add to list
        cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                              cam_intrinsics)
        args.cam_intrinsics.append(cam_intrinsics) # Add to list of intrinsics

        ### BOX (vs) BAXTER DATA
        if args.box_data:
            # Get ctrl dimension
            if args.ctrl_type == 'ballposforce':
                args.num_ctrl = 6
            elif args.ctrl_type == 'ballposeforce':
                args.num_ctrl = 10
            elif args.ctrl_type == 'ballposvelforce':
                args.num_ctrl = 9
            elif args.ctrl_type == 'ballposevelforce':
                args.num_ctrl = 13
            else:
                assert False, "Ctrl type unknown: {}".format(args.ctrl_type)
            print('Num ctrl: {}'.format(args.num_ctrl))
        else:
            # Compute extrinsics
            cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

            # Get dimensions of ctrl & state
            try:
                statelabels, ctrllabels, trackerlabels = data.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
                print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
            except:
                statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
                ctrllabels = statelabels  # Just use the labels
                trackerlabels = []
                print("Could not read statectrllabels file. Reverting to labels in statelabels file")
            #args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
            #print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
            args.num_ctrl = len(ctrllabels)
            print('Num ctrl: {}'.format(args.num_ctrl))

            # Find the IDs of the controlled joints in the state vector
            # We need this if we have state dimension > ctrl dimension and
            # if we need to choose the vals in the state vector for the control
            ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
            print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

            # Add to list of intrinsics
            args.cam_extrinsics.append(cam_extrinsics)
            args.ctrl_ids.append(ctrlids_in_state)
            args.state_labels.append(statelabels)

    # Data noise
    if not hasattr(args, "add_noise_data") or (len(args.add_noise_data) == 0):
        args.add_noise_data = [False for k in xrange(len(args.data))] # By default, no noise
    else:
        assert(len(args.data) == len(args.add_noise_data))
    if hasattr(args, "add_noise") and args.add_noise: # BWDs compatibility
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

    # Image suffix
    args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data (for baxter)
    if (not args.box_data):
        args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
        args.mesh_ids      = args.baxter_labels['meshIds']

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat', 'se3aar']), 'Unknown SE3 type: ' + args.se3_type
    args.delta_pivot = '' if (args.delta_pivot == 'None') else args.delta_pivot # Workaround since we can't specify empty string in the yaml
    assert (args.delta_pivot in ['', 'pred', 'ptmean', 'maskmean', 'maskmeannograd', 'posecenter']),\
        'Unknown delta pivot type: ' + args.delta_pivot
    delta_pivot_type = ' Delta pivot type: {}'.format(args.delta_pivot) if (args.delta_pivot != '') else ''
    #args.se3_dim       = ctrlnets.get_se3_dimension(args.se3_type)
    #args.delta_se3_dim = ctrlnets.get_se3_dimension(args.se3_type, (args.delta_pivot != '')) # Delta SE3 type
    if args.se3_type == 'se3aar':
        assert(args.delta_pivot == '')
    print('Predicting {} SE3s of type: {}.{}'.format(args.num_se3, args.se3_type, delta_pivot_type))

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
        args.loss_scale, args.pt_wt, args.consis_wt))

    # Weight sharpening stuff
    if args.use_wt_sharpening:
        print('Using weight sharpening to encourage binary mask prediction. Start iter: {}, Rate: {}, Noise stop iter: {}'.format(
            args.sharpen_start_iter, args.sharpen_rate, args.noise_stop_iter))

    # Normal loss
    if (args.normal_wt > 0):
        print('Using cosine similarity loss on the predicted normals. Loss wt: {}'.format(args.normal_wt))
        if args.bilateral_depth_smoothing:
            print('Applying bi-lateral filter to smooth the depths before computing normals.'
                  ' Window size: {}x{}, Depth std: {}'.format(args.bilateral_window_width, args.bilateral_window_width,
                                                              args.bilateral_depth_std))

    # Pose center anchor loss
    if (args.pose_anchor_wt > 0):
        print('Adding loss encouraging pose centers to be close to mask mean. Loss wt: {}'.format(args.pose_anchor_wt))
        if args.pose_anchor_maskbprop:
            print("Will backprop pose anchor error gradient to mask mean")

    # Loss type
    delta_loss = ', Penalizing the delta-flow loss per unroll'
    norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
    print('3D loss type: ' + args.loss_type + norm_motion + delta_loss)

    # Supervised segmentation loss
    if (args.seg_wt > 0):
        print("Using supervised segmentation loss with weight: {}".format(args.seg_wt))

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    # Box data
    if args.box_data:
        assert (not args.use_jt_angles), "Cannot use joint angles as input to the encoder for box data"
        assert (not args.use_jt_angles_trans), "Cannot use joint angles as input to the transition model for box data"
        assert (not args.reject_left_motion), "Cannot filter left arm motions for box data"
        assert (not args.reject_right_still), "Cannot filter right arm still cases for box data"

    if args.use_jt_angles:
        print("Using Jt angles as input to the pose encoder")

    if args.use_jt_angles_trans:
        print("Using Jt angles as input to the transition model")

    # DA threshold / winsize
    print("Flow/visibility computation. DA threshold: {}, DA winsize: {}".format(args.da_threshold,
                                                                                 args.da_winsize))
    if args.use_only_da_for_flows:
        print("Computing flows using only data-associations. Flows can only be computed for visible points")
    else:
        print("Computing flows using tracker poses. Can get flows for all input points")

    ########################
    ############ Load datasets
    # Get datasets
    load_color = None
    if args.use_xyzrgb:
        load_color = 'rgb'
    elif args.use_xyzhue:
        load_color = 'hsv'
    if args.reject_left_motion:
        print("Examples where any joint of the left arm moves by > 0.005 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.reject_right_still:
        print("Examples where no joint of the right arm move by > 0.015 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.add_noise:
        print("Adding noise to the depths, actual configs & ctrls")
    ### Box dataset (vs) Other options
    if args.box_data:
        print("Box dataset")
        valid_filter, args.mesh_ids = None, None # No valid filter
        read_seq_func = data.read_box_sequence_from_disk
    else:
        print("Baxter dataset")
        valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                                         mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                         reject_left_motion=args.reject_left_motion,
                                                                         reject_right_still=args.reject_right_still)
        read_seq_func = read_baxter_sequence_from_disk
    ### Noise function
    noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                     defprob=0.005, noisestd=0.005)
    ### Load functions
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per,
                                                         valid_filter = valid_filter,
                                                         cam_extrinsics=args.cam_extrinsics,
                                                         cam_intrinsics=args.cam_intrinsics,
                                                         ctrl_ids=args.ctrl_ids,
                                                         state_labels=args.state_labels,
                                                         add_noise=args.add_noise_data)
    disk_read_func  = lambda d, i: read_seq_func(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                 img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                 num_ctrl=args.num_ctrl,
                                                 mesh_ids = args.mesh_ids,
                                                 compute_bwdflows=args.use_gt_masks,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows,
                                                 noise_func=noise_func,
                                                 load_color=load_color,
                                                 compute_normals=(args.normal_wt > 0),
                                                 maxdepthdiff=args.normal_max_depth_diff,
                                                 bismooth_depths=args.bilateral_depth_smoothing,
                                                 bismooth_width=args.bilateral_window_width,
                                                 bismooth_std=args.bilateral_depth_std,
                                                 supervised_seg_loss=(args.seg_wt > 0)) # Need BWD flows / masks if using GT masks
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
    # train_sampler = torch.utils.data.dataloader.SequentialSampler(train_dataset)  # Run sequentially along the test dataset
    # train_loader  = DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                                               num_workers=args.num_workers, sampler=train_sampler,
    #                                               pin_memory=args.use_pin_memory,
    #                                               collate_fn=train_dataset.collate_batch))
    # val_sampler = torch.utils.data.dataloader.SequentialSampler(val_dataset)  # Run sequentially along the test dataset
    # val_loader  = DataEnumerator(util.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    #                                             num_workers=args.num_workers, sampler=val_sampler,
    #                                             pin_memory=args.use_pin_memory,
    #                                             collate_fn=val_dataset.collate_batch))
    # test_sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    # test_loader  = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                              num_workers=args.num_workers, sampler=test_sampler,
    #                                              pin_memory=args.use_pin_memory,
    #                                              collate_fn=test_dataset.collate_batch))
    ########################
    ############ Load models & optimization stuff

    assert not args.use_full_jt_angles, "Can only use as many jt angles as the control dimension"
    print('Using state of controllable joints')
    args.num_state_net = args.num_ctrl # Use only the jt angles of the controllable joints

    print('Using multi-step Flow-Model')
    if args.se2_data:
        print('Using the smaller multi-step SE2-Pose-Model')
    else:
        print('Using multi-step SE3-Pose-Model')

    ### Load the model
    num_train_iter = 0
    num_input_channels = 3 # Num input channels
    if args.use_xyzrgb:
        num_input_channels = 6
    elif args.use_xyzhue:
        num_input_channels = 4 # Use only hue as input
    if args.use_gt_masks:
        print('Using GT masks. Model predicts only poses & delta-poses')
        assert not args.use_gt_poses, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.MultiStepSE2OnlyPoseModel if args.se2_data else ctrlnets.MultiStepSE3OnlyPoseModel
    elif args.use_gt_poses:
        print('Using GT poses & delta poses. Model predicts only masks')
        assert not args.use_gt_masks, "Cannot set option for using GT masks and poses together"
        modelfn = se2nets.MultiStepSE2OnlyMaskModel if args.se2_data else ctrlnets.MultiStepSE3OnlyMaskModel
    else:
        modelfn = se2nets.MultiStepSE2PoseModel if args.se2_data else ctrlnets.MultiStepSE3PoseModel
    model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                    se3_type=args.se3_type, delta_pivot=args.delta_pivot, use_kinchain=False,
                    input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                    init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                    use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                    sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv, decomp_model=args.decomp_model,
                    use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                    wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                    use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state_net,
                    full_res=args.full_res, noise_stop_iter=args.noise_stop_iter,
                    trans_type=args.trans_type, posemask_type=args.posemask_type) # noise_stop_iter not available for SE2 models
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    # optionally resume from a checkpoint
    if args.resume:
        # TODO: Save path to TB log dir, save new log there again
        # TODO: Reuse options in args (see what all to use and what not)
        # TODO: Use same num train iters as the saved checkpoint
        # TODO: Print some stats on the training so far, reset best validation loss, best epoch etc
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint       = torch.load(args.resume)
            loadargs         = checkpoint['args']
            args.start_epoch = checkpoint['epoch']
            if args.reset_train_iter:
                num_train_iter   = 0 # Reset to 0
            else:
                num_train_iter   = checkpoint['train_iter']
            try:
                model.load_state_dict(checkpoint['state_dict']) # BWDs compatibility (TODO: remove)
            except:
                model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
                  .format(args.resume, checkpoint['epoch'], num_train_iter))
            best_loss    = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
            best_floss   = checkpoint['best_flow_loss'] if 'best_flow_loss' in checkpoint else float("inf")
            best_fcloss  = checkpoint['best_flowconsis_loss'] if 'best_flowconsis_loss' in checkpoint else float("inf")
            best_epoch   = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
            best_fepoch  = checkpoint['best_flow_epoch'] if 'best_flow_epoch' in checkpoint else 0
            best_fcepoch = checkpoint['best_flowconsis_epoch'] if 'best_flowconsis_epoch' in checkpoint else 0
            print('==== Best validation loss: {} was from epoch: {} ===='.format(best_loss, best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        assert(False)

    ######
    ## Iterate over train/val/test set and save the data
    datakeys = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}
    posedata = {}
    for key, val in datakeys.items():
        posedata[key] = iterate(val, model, key)

    # Save data
    torch.save(posedata, args.save_filename)

### Main iterate function (train/test/val)
def iterate(dataset, model, mode='test'):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time  = AverageMeter(), AverageMeter()

    # Switch model modes
    model.eval()

    # Read data
    gt_poses_d, pred_poses_d, jtangles_d = [], [], []
    dfids = {}

    # Iterate over all the examples
    print('========== Dataset: {}, Num iters: {} =========='.format(mode, len(dataset)))
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    total_start = time.time()
    for i in xrange(len(dataset)):
        # ============ Load data ============ #
        # Start timer
        start = time.time()

        # Get a sample
        sample = dataset[i] # Get sample
        if not sample['poses'].eq(sample['poses']).all():
            print("Sample has NaN in ground truth poses")
            continue

        # Get inputs and targets (as variables)
        pts      = util.to_var(sample['points'].type(deftype), volatile=True)

        # Get XYZRGB input
        if args.use_xyzrgb:
            rgb = util.to_var(sample['rgbs'].type(deftype)/255.0, requires_grad=False) # Normalize RGB to 0-1
            netinput = torch.cat([pts, rgb], 1) # Concat along channels dimension
        elif args.use_xyzhue:
            hue = util.to_var(sample['rgbs'].narrow(2,0,1).type(deftype)/179.0, requires_grad=False)  # Normalize Hue to 0-1 (Opencv has hue from 0-179)
            netinput = torch.cat([pts, hue], 1) # Concat along channels dimension
        else:
            netinput = pts # XYZ

        # Get jt angles
        jtangles = util.to_var(sample['actctrlconfigs'].type(deftype), requires_grad=False) #[:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============ #
        # Start timer
        start = time.time()

        ### Predict the pose from the network
        pred_poses = model.forward_only_pose([netinput[0:1], jtangles[0:1]])

        # Measure fwd pass time
        fwd_time.update(time.time() - start)

        # ============ Save stuff ============ #
        did = sample['datasetid']  # Dataset ID
        fid = sample['folderid']  # Folder ID in dataset
        if (did, fid) not in dfids:
            print('Added new ID ({}: {}, {})'.format(len(dfids), did, fid))
            dfids[(did, fid)] = len(dfids)  # We have seen this pair
            pred_poses_d.append([])
            gt_poses_d.append([])
            jtangles_d.append([])
        # Save to list
        pred_poses_d[-1].append(pred_poses.data.cpu().clone())
        gt_poses_d[-1].append(sample['poses'][0:1].clone()) # Choose one element only, 1 x 8 x 3 x 4
        jtangles_d[-1].append(sample['actctrlconfigs'][0:1].clone()) # 1 x 7

        # Print stats
        if i % 5000 == 0:
            print('Dataset: {}, Data-Folder ID: {}, Example: {}/{}'.format(mode, len(pred_poses_d), i+1, len(dataset)))
            print('\tTime => Total: {:.3f}, Data: {data.val:.3f} ({data.avg:.3f}), '
                  'Fwd: {fwd.val:.3f} ({fwd.avg:.3f})'.format(
                time.time() - total_start, data=data_time, fwd=fwd_time))

    ## Concat stuff and return
    for kk in xrange(len(gt_poses_d)):
        gt_poses_d[kk]   = torch.cat(gt_poses_d[kk], 0)    # N x 8 x 3 x 4
        pred_poses_d[kk] = torch.cat(pred_poses_d[kk], 0)  # N x 8 x 3 x 4
        jtangles_d[kk] = torch.cat(jtangles_d[kk], 0)      # N x 7
    return {'gtposes': gt_poses_d, 'predposes': pred_poses_d, 'jtangles': jtangles_d}


################ RUN MAIN
if __name__ == '__main__':
    main()
