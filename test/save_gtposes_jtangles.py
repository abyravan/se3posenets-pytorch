# Torch imports
import torch
import torch.optim
import torch.utils.data

# Numpy
import numpy as np

# Local imports
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import data
import util

#### Setup options
# Common
import options
parser = options.setup_comon_options()

# Loss options
parser.add_argument('--pt-wt', default=1, type=float,
                    metavar='WT', help='Weight for the 3D point loss - only FWD direction (default: 1)')
parser.add_argument('--save-filename', default='gtposedata-sim.pth.tar', type=str,
                    metavar='PATH', help='file name to save results in (default: gtposedata-sim.pth.tar)')

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ Stripped out function to read data from disk
### Generate the data files (with all the depth, flow etc.) for each sequence
def generate_baxter_sequence(dataset, idx):
    # Get stuff from the dataset
    step, seq, suffix = dataset['step'], dataset['seq'], dataset['suffix']
    # If the dataset has subdirs, find the proper sub-directory to use
    did = 0
    if ('subdirs' in dataset) :#dataset.has_key('subdirs')):
        # Find the sub-directory the data falls into
        assert (idx < dataset['numdata']);  # Check if we are within limits
        did = np.searchsorted(dataset['subdirs']['datahist'], idx, 'right') - 1  # ID of sub-directory. If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID
        # Update the ID and path so that we get the correct images
        id   = idx - dataset['subdirs']['datahist'][did] # ID of image within the sub-directory
        path = dataset['path'] + '/' + dataset['subdirs']['dirnames'][did] + '/' # Get the path of the sub-directory
    else:
        id   = dataset['ids'][idx] # Select from the list of valid ids
        path = dataset['path'] # Root of dataset
    # Setup start/end IDs of the sequence
    start, end = id, id + (step * seq)
    sequence, ct, stepid = {}, 0, step
    for k in xrange(start, end + 1, step):
        sequence[ct] = {'depth'     : path + 'depth' + suffix + str(k) + '.png',
                        'label'     : path + 'labels' + suffix + str(k) + '.png',
                        'color'     : path + 'color' + suffix + str(k) + '.png',
                        'state1'    : path + 'state' + str(k) + '.txt',
                        'state2'    : path + 'state' + str(k + 1) + '.txt',
                        'se3state1' : path + 'se3state' + str(k) + '.txt',
                        'se3state2' : path + 'se3state' + str(k + 1) + '.txt',
                        'flow'   : path + 'flow_' + str(stepid) + '/flow' + suffix + str(start) + '.png',
                        'visible': path + 'flow_' + str(stepid) + '/visible' + suffix + str(start) + '.png'}
        stepid += step  # Get flow from start image to the next step
        ct += 1  # Increment counter
    return sequence, path, did

### Load baxter sequence from disk
def read_gtposesctrls_from_disk(dataset, id,
                                ctrl_type='actdiffvel', num_ctrl=7,
                                mesh_ids=torch.Tensor()):
    # Setup vars
    num_meshes = mesh_ids.nelement()  # Num meshes
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics, camera_extrinsics, ctrl_ids = dataset['camintrinsics'], dataset['camextrinsics'], dataset['ctrlids']

    # Setup memory
    sequence, path, folid = data.generate_baxter_sequence(dataset, id)  # Get the file paths
    actctrlconfigs = torch.FloatTensor(seq_len + 1, num_ctrl)  # Ids in actual data belonging to commanded data
    poses = torch.FloatTensor(seq_len + 1, mesh_ids.nelement() + 1, 3, 4).zero_()

    # For computing FWD/BWD visibilities and FWD/BWD flows
    allposes = torch.FloatTensor()

    ## Read camera extrinsics (can be separate per dataset now!)
    try:
        camera_extrinsics = data.read_cameradata_file(path + '/cameradata.txt')
    except:
        pass  # Can use default cam extrinsics for the entire dataset

    #####
    # Load sequence
    t = torch.linspace(0, seq_len * step_len * (1.0 / 30.0), seq_len + 1).view(seq_len + 1, 1)  # time stamp
    for k in xrange(len(sequence)):
        # Get data table
        s = sequence[k]

        # Load configs
        state = data.read_baxter_state_file(s['state1'])
        actctrlconfigs[k] = state['actjtpos'][ctrl_ids]  # Get states for control IDs
        if state['timestamp'] is not None:
            t[k] = state['timestamp']

        # Load SE3 state & get all poses
        se3state = data.read_baxter_se3state_file(s['se3state1'])
        if allposes.nelement() == 0:
            allposes.resize_(seq_len + 1, len(se3state) + 1, 3, 4).fill_(0)  # Setup size
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

    # Different control types
    dt = t[1:] - t[:-1]  # Get proper dt which can vary between consecutive frames
    controls = (actctrlconfigs[1:seq_len + 1] - actctrlconfigs[0:seq_len]) / dt  # state -> ctrl dimension

    # Return loaded data
    dataout = {'controls': controls, 'poses': poses, 'folderid': folid,
            'dt': dt, 'actctrlconfigs': actctrlconfigs}
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

        ### BAXTER DATA
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
    args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat', 'se3aar']), 'Unknown SE3 type: ' + args.se3_type

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
        args.loss_scale, args.pt_wt, args.consis_wt))

    # Weight sharpening stuff
    if args.use_wt_sharpening:
        print('Using weight sharpening to encourage binary mask prediction. Start iter: {}, Rate: {}, Noise stop iter: {}'.format(
            args.sharpen_start_iter, args.sharpen_rate, args.noise_stop_iter))

    # Loss type
    delta_loss = ', Penalizing the delta-flow loss per unroll'
    norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
    print('3D loss type: ' + args.loss_type + norm_motion + delta_loss)

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

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

    ### Baxter dataset
    print("Baxter dataset")
    valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                                     mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                     reject_left_motion=args.reject_left_motion,
                                                                     reject_right_still=args.reject_right_still)
    read_seq_func = read_gtposesctrls_from_disk

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
    disk_read_func  = lambda d, i: read_seq_func(d, i, ctrl_type = args.ctrl_type,
                                                 num_ctrl=args.num_ctrl,
                                                 mesh_ids = args.mesh_ids) # Need BWD flows / masks if using GT masks
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # Get train, test, val dataset elements
    datakeys = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}
    posedata = {}
    for key, val in datakeys.items():
        # Read data
        poses, jtangles = {}, {}
        for k in xrange(len(val)):
            sample = val[k] # Get sample
            did   = sample['datasetid'] # Dataset ID
            fid   = sample['folderid'] # Folder ID in dataset
            if did not in poses:
               poses[did], jtangles[did] = {}, {}
            if fid not in poses[did]:
                poses[did][fid], jtangles[did][fid] = [], []
            if k % 500 == 0:
                print('Dataset: {}/{}, Example: {}/{}'.format(key, did, k+1, len(val)))
            if sample['poses'].eq(sample['poses']).all(): # Only if there are no NaN values
                poses[did][fid].append(sample['poses'][0:1]) # Choose first element only, 1 x 8 x 3 x 4
                jtangles[did][fid].append(sample['actctrlconfigs'][0:1]) # 1 x 7
                # ctrls[did].append(sample['controls'][0:1])
        # Save data
        for kk, _ in poses.items():
            for jj, _ in poses[kk].items():
                poses[kk][jj] = torch.cat(poses[kk][jj], 0) # N x 8 x 3 x 4
                jtangles[kk][jj] = torch.cat(jtangles[kk][jj], 0) # N x 8 x 3 x 4
        posedata[key] = {'gtposes': poses,
                         #'controls': torch.cat(ctrls,0),
                         'jtangles': jtangles}

    # Save data
    torch.save(posedata, args.save_filename)

################ RUN MAIN
if __name__ == '__main__':
    main()

