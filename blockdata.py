# General imports
import h5py
import os
import numpy as np
import io
from PIL import Image

# Torch imports
import torch
import torch.utils.data

# Local import
import se3
import data

############
##### Helper functions for loading images
def RGBToDepth(img):
    return img[:,:,0]+.01*img[:,:,1]+.0001*img[:,:,2]

def RGBAToMask(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        mask += np.left_shift(buf[:,:, dim], shift)
    return mask

def RGBAArrayToMasks(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        mask += np.left_shift(buf[:,:,:, dim], shift)
    return mask

def PNGToNumpy(png):
    stream = io.BytesIO(png)
    im = Image.open(stream)
    return np.asarray(im, dtype=np.uint8)

def ConvertPNGListToNumpy(inputs):
    imgs = []
    for raw in inputs:
        imgs.append(PNGToNumpy(raw))
    arr = np.array(imgs)
    return arr

def ConvertDepthPNGListToNumpy(inputs):
    imgs = []
    for raw in inputs:
        imgs.append(RGBToDepth(PNGToNumpy(raw)))
    arr = np.array(imgs)
    return arr

def NumpyBHWCToTorchBCHW(array):
    return torch.from_numpy(array).permute(0,3,1,2).contiguous()

def NumpyBHWToTorchBCHW(array):
    return torch.from_numpy(array).unsqueeze(1).contiguous()

# Rotation about the Y-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
def create_roty_np(theta):
    rot = np.eye(4,4)
    rot[0, 0] = np.cos(theta)
    rot[2, 2] = rot[0, 0]
    rot[2, 0] = np.sin(theta)
    rot[0, 2] = -rot[2, 0]
    return rot

# Rotation about the Z-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
def create_rotz_np(theta):
    rot = np.eye(4,4)
    rot[0, 0] = np.cos(theta)
    rot[1, 1] = rot[0, 0]
    rot[0, 1] = np.sin(theta)
    rot[1, 0] = -rot[0, 1]
    return rot

############
##### Helper functions for reading the data directories & loading train/test files
def read_block_sim_dataset(load_dirs, step_len, seq_len, train_per=0.6, val_per=0.15,
                           use_failures=True, remove_static_examples=False):
    # Get all the load directories
    assert (train_per + val_per <= 1)  # Train + val + test <= 1

    # Iterate over each load directory to find the datasets
    datasets = []
    for load_dir in load_dirs:
        # Get h5 file names & number of examples
        files = os.listdir(load_dir)
        filenames, numvaliddata, validdataids, numdata = [], [], [], 0
        for file in files:
            # Discard non-valid files
            if file.find('.h5') == -1:
                continue # Skip non h5 files
            if (not use_failures) and (file.find('failure') != -1):
                continue # Skip failures

            # Read number of example images in the file
            max_flow_step = int(step_len * seq_len)  # This is the maximum future step (k) for which we need flows
            with h5py.File(os.path.join(load_dir, file), 'r') as h5data:
                # Get number of examples from that h5
                nexamples = len(h5data['images_rgb']) - max_flow_step  # We only have flows for these many images!
                if (nexamples < 1):
                    continue

                # Get the joint angles and filter examples where the joints don't move
                # This function checks all examples "apriori" to see if they are valid
                # and returns a set of ids such that the sequence of examples from that id
                # to id + seq*step are valid
                if remove_static_examples:
                    validids = []
                    arm_r_idx = [1, 3, 5, 7, 9, 11, 13]  # No gripper (right)
                    jtstates  = torch.from_numpy(h5data['robot_positions'][:, arm_r_idx]).float()
                    for k in range(nexamples):
                        st, ed = k, k + int(step_len * seq_len)
                        seq = list(np.arange(st, ed + 1, step_len))
                        statediff = (jtstates[seq[1:]] - jtstates[seq[:-1]]).abs().mean(1).gt(1e-3)
                        if statediff.sum() < seq_len-1: # Allow for one near zero example in sequence
                            continue
                        validids.append(k) # Accept this example
                else:
                    validids = range(0, nexamples)  # Is just the same as ids, all samples are valid

                # Valid training motion
                numdata += nexamples
                numvaliddata.append(len(validids))
                validdataids.append(validids)
                filenames.append(file)

        # Print stats
        print('Found {}/{} valid motions ({}/{}, {}% valid examples) in dataset: {}'.format(len(numvaliddata), len(files),
                       sum(numvaliddata), numdata, sum(numvaliddata) * (1.0/numdata), load_dir))

        # Setup training and test splits in the dataset, here we actually split based on the h5s
        nfiles = len(filenames)
        nfilestrain, nfilesval = int(train_per * nfiles), int(val_per * nfiles) # First train_per datasets for training, next val_per for validation
        nfilestest = int(nfiles - (nfilestrain + nfilesval)) # Rest files are for testing

        # Get number of images in the datasets
        nvalidexamples = sum(numvaliddata)
        ntrain = sum(numvaliddata[:nfilestrain]) # Num images for training
        nval   = sum(numvaliddata[nfilestrain:nfilestrain+nfilesval]) # Validation
        ntest  = nvalidexamples - (ntrain + nval) # Number of test images
        print('\tNum train: {} ({}), val: {} ({}), test: {} ({})'.format(
            nfilestrain, ntrain, nfilesval, nval, nfilestest, ntest))

        # Setup the dataset structure
        numvaliddata.insert(0, 0)  # Add a zero in front for the cumsum
        dataset = {'path'   : load_dir,
                   'step'   : step_len,
                   'seq'    : seq_len,
                   'numdata': nvalidexamples,
                   'train'  : [0, ntrain - 1],
                   'val'    : [ntrain, ntrain + nval - 1],
                   'test'   : [ntrain + nval, nvalidexamples - 1],
                   'files'  : {'names'   : filenames,
                               'ids'     : validdataids,
                               'datahist': np.cumsum(numvaliddata),
                               'train'   : [0, nfilestrain - 1],
                               'val'     : [nfilestrain, nfilestrain + nfilesval - 1],
                               'test'    : [nfilestrain + nfilesval, nfiles - 1]},
                   }

        ##### Setup camera intrinsics and extrinsics
        ##### ASSUME: Intrinsics and extrinsics are the same for a given dataset directory
        # Load a single h5 example
        with h5py.File(os.path.join(load_dir, dataset['files']['names'][0]), 'r') as h5data:
            # Load a single RGB image
            rgb = np.array(PNGToNumpy(h5data['images_rgb'][0])) / 255. #

            # Figure out camera intrinsics
            img_height, img_width = rgb.shape[0], rgb.shape[1]
            vfov = (np.pi / 180.) * float(np.array(h5data['camera_fov']))
            fx = fy = img_height / (2 * np.tan(vfov / 2))
            cx, cy = 0.5 * img_width, 0.5 * img_height
            camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                                 'width': img_width, 'height': img_height}

            # Compute xygrid and add to intrinsics
            camera_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(img_height, img_width,
                                                                                     camera_intrinsics)

            # Figure out camera extrinsics
            # Rotate the modelview from the simulated datasets to match with our assumptions
            # of x to right and y down and z pointed forward in camera frame.
            modelview = np.array(h5data['camera_view_matrix']).reshape(4, 4).transpose()  # Global to Camera transform
            roty_pi, rotz_pi = create_roty_np(np.pi), create_rotz_np(np.pi)
            modelview_c = rotz_pi.dot(roty_pi.dot(modelview)) # rotz_pi * roty_pi * modelview
            camera_extrinsics = {'modelView': torch.from_numpy(modelview_c).clone().float()}

            # Add to dataset
            dataset['camera_intrinsics'] = camera_intrinsics
            dataset['camera_extrinsics'] = camera_extrinsics

        # Append to list of all datasets
        datasets.append(dataset)

    # Return
    return datasets

##### Generate the data files (with all the depth, flow etc.) for each sequence
def generate_block_sequence(dataset, idx):
    # Get stuff from the dataset
    step, seq = dataset['step'], dataset['seq']
    # If the dataset has files, find the proper file to use
    did = 0
    if ('files' in dataset):
        # Find the sub-directory the data falls into
        assert (idx < dataset['numdata'])  # Check if we are within limits
        did = np.searchsorted(dataset['files']['datahist'], idx, 'right') - 1  # ID of file. If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID
        # Update the ID and path so that we get the correct images
        id   = idx - dataset['files']['datahist'][did] # ID of "first" image within the file
        path = dataset['path'] + '/' + dataset['files']['names'][did] # Get the path of the file
        # Valid ID
        vid = dataset['files']['ids'][did][id] # Start ID (based on valid ids, if all are valid, it will be = id, else different)
        st, ed = vid, vid + (step * seq)
    else:
        assert(False) # TODO: For real data, check this to make sure things are right
        id   = dataset['ids'][idx] # Select from the list of valid ids
        path = dataset['path'] # Root of dataset
        st, ed = id, id + (step * seq)

    # Setup start/end IDs of the sequence
    sequence = list(np.arange(st, ed+1, step))
    return sequence, path, int(did)

##### Load block sequence from disk
def read_block_sequence_from_disk(dataset, id, ctrl_type='actdiffvel', robot='yumi',
                                  gripper_ctrl_type = 'vel', compute_bwdflows=True, load_color=False,
                                  dathreshold=0.01, dawinsize=5, use_only_da=False,
                                  noise_func=None):
    # Setup vars
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics, camera_extrinsics = dataset['camera_intrinsics'], \
                                           dataset['camera_extrinsics']

    # Setup memory
    seq, path, fileid = generate_block_sequence(dataset, id)  # Get the file paths

    ### Load data from the h5 file
    # Get depth, RGB, labels and poses
    with h5py.File(path, 'r') as h5data:
        ##### Read image data
        # Get RGB images
        rgbs = None
        if load_color:
            rgbs = NumpyBHWCToTorchBCHW(ConvertPNGListToNumpy(h5data['images_rgb'][seq]))

        # Get depth images
        depths = NumpyBHWToTorchBCHW(ConvertDepthPNGListToNumpy(h5data['images_depth'][seq])).float()

        # Get segmentation label images (numpy)
        labels  = RGBAArrayToMasks(ConvertPNGListToNumpy(h5data['images_mask'][seq]))

        ##### Get object poses
        # Find unique object IDs and poses (assumes quaternion representation)
        # Make sure that the BG poses are initialized to identity
        poses, objids = [], {}
        for key in h5data.keys():
            if (key.find("pose") != -1):
                if (h5data[key].shape[0] != 0):
                    poses.append(np.array(h5data[key])[seq])
                else:
                    poses.append(np.zeros((len(seq), 7))) # Empty array
                    poses[-1][:,-1] = 1. # Unit quaternion for identity rotation
                objids[int(key[4:])] = len(objids)
        poses = torch.Tensor(poses).permute(1,0,2).clone().float() # Setup poses (S x NPOSES X 7)

        # Convert quaternions to rotation matrices (basically get 3x4 matrices)
        rtposes = se3.SE3ToRt(poses, 'se3quat', False)

        # Transform all points from the object's local frame of reference to camera frame
        rtposes_cam = se3.ComposeRtPair(camera_extrinsics['modelView'][:-1].view(1,1,3,4).expand_as(rtposes),
                                        rtposes)

        ##### Relabel segmentation mask to agree with the new pose ids
        # remap mask values
        labels_remapped = np.copy(labels)
        for ky, vl in objids.items():
            labels_remapped[labels == ky] = vl # Remap to values from 0 -> num_poses
        labels_remapped = NumpyBHWToTorchBCHW(labels_remapped).byte()

        ##### Get joint state and controls
        dt = step_len * (1.0/30.0)
        if robot == 'yumi':
            # Indices for extracting current arm position (excluding gripper). When
            # computing the state and controls for YUMI robot.
            arm_l_idx = [0, 2, 4, 6, 8, 10, 12] #, 15] # Last ID is gripper (left)
            arm_r_idx = [1, 3, 5, 7, 9, 11, 13] #, 14] # Last ID is gripper (right)

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

            # # Gripper control
            # if gripper_ctrl_type == 'vel':
            #     pass # This is what we have already
            # elif gripper_ctrl_type == 'compos':
            #     ming, maxg = 0.015, 0.025 # Min/Max gripper positions
            #     gripper_cmds = (torch.from_numpy(h5data['right_gripper_cmd'][seq[:-1]]) - ming) / (maxg - ming)
            #     gripper_cmds.clamp_(0,1) # Normalize and clamp to 0/1
            #     controls[:,-1] = gripper_cmds # Update the gripper controls to be the actual position commands
            # else:
            #     assert False, "Unknown gripper control type input for the YUMI: {}".format(gripper_ctrl_type)
        else:
            assert False, "Unknown robot type input: {}".format(robot)

        ##### Compute 3D point cloud from depth using camera intrinsics
        # Add noise to the depths before we compute the point cloud
        if (noise_func is not None):
            depths_n = noise_func(depths)
            depths.copy_(depths_n)  # Replace by noisy depths

        # Compute x & y values for the 3D points (= xygrid * depths)
        points = torch.zeros(seq_len+1, 3, camera_intrinsics['height'], camera_intrinsics['width'])
        xy = points[:, 0:2]
        xy.copy_(camera_intrinsics['xygrid'].expand_as(xy))  # = xygrid
        xy.mul_(depths.expand(seq_len + 1, 2, camera_intrinsics['height'], camera_intrinsics['width'])) # = xy * z
        points[:, 2:].copy_(depths) # Copy depths to 3D points

        # Compute the flows and visibility
        tarpts = points[1:]  # t+1, t+2, t+3, ....
        initpt = points[0:1].expand_as(tarpts)
        tarlabels = labels_remapped[1:]  # t+1, t+2, t+3, ....
        initlabel = labels_remapped[0:1].expand_as(tarlabels)
        tarposes = rtposes_cam[1:]  # t+1, t+2, t+3, ....
        initpose = rtposes_cam[0:1].expand_as(tarposes)

        # Compute flow and visibility
        fwdflows, bwdflows, \
        fwdvisibilities, bwdvisibilities, \
        fwdassocpixelids, bwdassocpixelids = data.ComputeFlowAndVisibility(initpt, tarpts, initlabel, tarlabels,
                                                                           initpose, tarposes, camera_intrinsics,
                                                                           dathreshold, dawinsize, use_only_da)

    # Return loaded data
    output = {'points': points, 'fwdflows': fwdflows, 'fwdvisibilities': fwdvisibilities,
              'fwdassocpixelids': fwdassocpixelids, 'controls': controls, 'states': states,
              'labels': labels_remapped, 'poses': rtposes_cam, 'dt': dt, 'fileid': int(fileid)}
    if compute_bwdflows:
        output['bwdflows'] = bwdflows
        output['bwdvisibilities'] = bwdvisibilities
        output['bwdassocpixelids'] = bwdassocpixelids
    if load_color:
        output['rgbs'] = rgbs

    return output

#####
def filter_func(batch, mean_dt, std_dt):
    # Check if there are any nans in the sampled poses. If there are, then discard the sample
    filtered_batch = []
    for sample in batch:
        # Check if any dt is too large (within 2 std. deviations of the mean)
        tok = ((sample['dt'] - mean_dt).abs_() < 2 * std_dt).all()
        # Check if there are NaNs in the poses
        poseok = sample['poses'].eq(sample['poses']).all()
        # Append if both checks pass
        if tok and poseok:
            filtered_batch.append(sample)
    # Return
    return filtered_batch

###################### DATASET
##### Dataset for Block Sequences
class BlockSeqDataset(torch.utils.data.Dataset):
    ''' Datasets for training SE3-Pose-Nets based on Block Sequential data '''

    def __init__(self, datasets, load_function, dtype='train', filter_func=None):
        '''
        Create the data loader given paths to existing list of datasets:
        :param datasets: 		List of datasets that have train | test | val splits
        :param load_function:	Function for reading data from disk given a dataset and an ID (this function needs to
                                return a dictionary of torch tensors)
        :param dtype:			Type of dataset: 'train', 'test' or 'val'
        :param filter_func:     Function that filters out bad samples from a batch during collating
        '''
        assert (len(datasets) > 0);  # Need atleast one dataset
        assert (dtype == 'train' or dtype == 'val' or dtype == 'test')  # Has to be one of the types
        self.datasets = datasets
        self.load_function = load_function
        self.dtype = dtype
        self.filter_func = filter_func  # Filters samples in the collater

        # Get some stats
        self.numdata = 0
        self.datahist = [0]
        for d in self.datasets:
            numcurrdata = int(d[self.dtype][1] - d[self.dtype][0] + 1)
            self.numdata += numcurrdata
            self.datahist.append(self.datahist[-1] + numcurrdata)
        print('Setting up {} dataset. Total number of data samples: {}'.format(self.dtype, self.numdata))

    def __len__(self):
        return self.numdata

    def __getitem__(self, idx):
        # Find which dataset to sample from
        assert (idx < self.numdata);  # Check if we are within limits
        did = np.digitize(idx, self.datahist) - 1  # If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID

        # Find ID of sample in that dataset (not the same as idx as we might have multiple datasets)
        start = self.datasets[did][self.dtype][0]  # This is the ID of the starting sample of the train/test/val part in the entire dataset
        diff = (idx - self.datahist[did])  # This will be from 0 - size for either train/test/val part of that dataset
        sid = int(start + diff)

        # Call the disk load function
        # Assumption: This function returns a dict of torch tensors
        sample = self.load_function(self.datasets[did], sid)
        sample['id'] = int(idx)  # Add the ID of the sample in
        sample['datasetid'] = int(did)  # Add the ID of the dataset in

        # Return
        return sample

    ### Collate the batch together
    def collate_batch(self, batch):
        # Filter batch based on custom function
        if self.filter_func is not None:
            filtered_batch = self.filter_func(batch)
        else:
            filtered_batch = batch

        # Collate the other samples together using the default collate function
        collated_batch = torch.utils.data.dataloader.default_collate(filtered_batch)

        # Return post-processed batch
        return collated_batch

###################### DATA LOADER SETUP
##### Setup the data loaders for the block datasets
def parse_options_and_setup_block_dataset_loader(args):
    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat',
                              'se3aar']), 'Unknown SE3 type: ' + args.se3_type

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
        args.loss_scale, args.pt_wt, args.consis_wt))

    # Weight sharpening stuff
    if args.use_wt_sharpening:
        print('Using weight sharpening to encourage binary mask prediction. '
              'Start iter: {}, Rate: {}, Noise stop iter: {}'.format(
              args.sharpen_start_iter, args.sharpen_rate, args.noise_stop_iter))

    # Loss type
    norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
    print('3D loss type: ' + args.loss_type + norm_motion)

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    if args.use_jt_angles:
        print("Using Jt angles as input to the pose encoder")

    # SE3NN model
    if args.use_se3nn:
        print('Using SE3NNs SE3ToRt layer implementation')
    else:
        print('Using the SE3ToRt implementation in se3.py')

    # DA threshold / winsize
    print("Flow/visibility computation. DA threshold: {}, DA winsize: {}".format(args.da_threshold,
                                                                                 args.da_winsize))
    if args.use_only_da_for_flows:
        print("Computing flows using only data-associations. Flows can only be computed for visible points")
    else:
        print("Computing flows using tracker poses. Can get flows for all input points")

    # YUMI robot
    if args.robot == "yumi":
        args.num_ctrl = 7 # Exclude the gripper now
        args.img_ht, args.img_wd = 240, 320
        print("Img ht: {}, Img wd: {}, Num ctrl: {}".format(args.img_ht, args.img_wd, args.num_ctrl))
    else:
        assert False, "Unknown robot type input: {}".format(args.robot)

    ########################
    ############ Load datasets
    # XYZ-RGB
    load_color = args.use_xyzrgb
    if args.use_xyzrgb:
        print("Using XYZ-RGB input - 6 channels. Assumes registered depth/RGB")

    # Noise addition
    noise_func = None
    if args.add_noise:
        print("Adding noise to the depths")
        noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                         defprob=0.005, noisestd=0.005)

    # Validity checker
    if args.remove_static_examples:
        print('Removing examples where the arm is static (this also excludes examples with just gripper motion)')

    ### Load functions
    block_data = read_block_sim_dataset(args.data,
                                        step_len=args.step_len,
                                        seq_len=args.seq_len,
                                        train_per=args.train_per,
                                        val_per=args.val_per,
                                        use_failures=args.use_failures,
                                        remove_static_examples=args.remove_static_examples)
    disk_read_func = lambda d, i: read_block_sequence_from_disk(d, i, ctrl_type=args.ctrl_type, robot=args.robot,
                                                                gripper_ctrl_type=args.gripper_ctrl_type,
                                                                compute_bwdflows=False,
                                                                dathreshold=args.da_threshold,
                                                                dawinsize=args.da_winsize,
                                                                use_only_da=args.use_only_da_for_flows,
                                                                noise_func=noise_func,
                                                                load_color=load_color)
    train_dataset = BlockSeqDataset(block_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = BlockSeqDataset(block_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = BlockSeqDataset(block_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset),
                                                                       len(val_dataset),
                                                                       len(test_dataset)))

    # Return
    return train_dataset, val_dataset, test_dataset


################ TEST
if __name__ == '__main__':

    ########
    import cv2

    # Project a 3D point to an image using the pinhole camera model (perspective transform)
    # Given a camera matrix of the form [fx 0 cx; 0 fy cy; 0 0 1] (x = cols, y = rows) and a 3D point (x,y,z)
    # We do: x' = x/z, y' = y/z, [px; py] = cameraMatrix * [x'; y'; 1]
    # Returns a 2D pixel (px, py)
    def project_to_image(camera_intrinsics, point):
        # Project to (0,0,0) if z = 0
        pointv = point.view(4)  # 3D point
        if pointv[2] == 0:
            return torch.zeros(2).type_as(point)

        # Perspective projection
        c = camera_intrinsics['fx'] * (pointv[0] / pointv[2]) + camera_intrinsics['cx']  # fx * (x/z) + cx
        r = camera_intrinsics['fy'] * (pointv[1] / pointv[2]) + camera_intrinsics['cy']  # fy * (y/z) + cy
        return torch.Tensor([c, r]).type_as(point)

    # Transform a point through the given pose (point in pose's frame of reference to global frame of reference)
    # Pose: (3x4 matrix) [R | t]
    # Point: (position) == torch.Tensor(3)
    # Returns: R*p + t == torch.Tensor(3)
    def transform(pose, point):
        pt = torch.mm(pose.view(4, 4), point.view(4, 1))
        return pt
        # posev, pointv = pose.view(3,4), point.view(3,1)
        # return torch.mm(posev[:,0:3], pointv).view(3) + posev[:,3] # R*p + t

    # Plot a 3d frame (X,Y,Z axes) of an object on a qt window
    # given the 6d pose of the object (3x4 matrix) in the camera frame of reference,
    # and the camera's projection matrix (3x3 matrix of form [fx 0 cx; 0 fy cy; 0 0 1])
    # Img represented as H x W x 3 (numpy array) & Pose is a 3 x 4 torch tensor
    def draw_3d_frame(img, pose, camera_intrinsics={}, pixlength=10.0, thickness=2):
        # Project the principal vectors (3 columns which denote the {X,Y,Z} vectors of the object) into the global (camera frame)
        dv = 0.2  # Length of 3D vector
        poset = torch.eye(4); poset[:3] = pose
        X = transform(poset, torch.FloatTensor([dv, 0, 0, 1]))
        Y = transform(poset, torch.FloatTensor([0, dv, 0, 1]))
        Z = transform(poset, torch.FloatTensor([0, 0, dv, 1]))
        O = transform(poset, torch.FloatTensor([0, 0, 0, 1]))
        # Project the end-points of the vectors and the frame origin to the image to get the corresponding pixels
        Xp = project_to_image(camera_intrinsics, X)
        Yp = project_to_image(camera_intrinsics, Y)
        Zp = project_to_image(camera_intrinsics, Z)
        Op = project_to_image(camera_intrinsics, O)
        # Maintain a specific length in pixel space by changing the tips of the frames to match correspondingly
        unitdirX = (Xp - Op).div_((Xp - Op).norm(2) + 1e-12)  # Normalize it
        unitdirY = (Yp - Op).div_((Yp - Op).norm(2) + 1e-12)  # Normalize it
        unitdirZ = (Zp - Op).div_((Zp - Op).norm(2) + 1e-12)  # Normalize it
        Xp = Op + pixlength * unitdirX
        Yp = Op + pixlength * unitdirY
        Zp = Op + pixlength * unitdirZ
        # Draw lines on the image
        cv2.line(img, tuple(Op.numpy()), tuple(Xp.numpy()), [1, 0, 0], thickness)
        cv2.line(img, tuple(Op.numpy()), tuple(Yp.numpy()), [0, 1, 0], thickness)
        cv2.line(img, tuple(Op.numpy()), tuple(Zp.numpy()), [0, 0, 1], thickness)

    ### Normalize image
    def normalize_img(img, min=-0.01, max=0.01):
        return (img - min) / (max - min)

    # Get data directories
    import sys
    import argparse
    args = argparse.Namespace()
    args.data = sys.argv[1].split(',') # Get inut data directories

    # Setup other options
    args.se3_type = 'se3aa'
    args.use_wt_sharpening = False
    args.loss_type, args.motion_norm_loss = 'mse', True
    args.step_len, args.seq_len = 2, 2
    args.loss_scale, args.pt_wt, args.consis_wt = 1.0, 1.0, 1.0
    args.wide_model = True
    args.use_se3nn = True
    args.use_jt_angles = True
    args.da_threshold = 0.015
    args.da_winsize = 5
    args.use_only_da_for_flows = False
    args.use_xyzrgb = True
    args.add_noise = False
    args.train_per, args.val_per = 0.6, 0.15
    args.use_failures = False
    args.remove_static_examples = True
    args.ctrl_type, args.robot, args.gripper_ctrl_type = 'actdiffvel', 'yumi', 'compos'

    # Setup datasets
    train_dataset, val_dataset, test_dataset = parse_options_and_setup_block_dataset_loader(args)

    # Load examples and visualize them
    import matplotlib.pyplot as plt

    # Display the data
    plt.ion()
    fig = plt.figure(100)

    # Render
    nimgs = 500
    for k in range(0, nimgs, 4):
        # Get sample
        sample = train_dataset[k]
        rgbs, poses = sample['rgbs'], sample['poses']

        # Project poses onto RGB images
        rgb1, rgb2 = sample['rgbs'][0].permute(1,2,0).clone().numpy() / 255.0, \
                     sample['rgbs'][1].permute(1,2,0).clone().numpy() / 255.0
        depth1, depth2 = sample['points'][0,2].clone().numpy(), \
                         sample['points'][1,2].clone().numpy()
        mask1, mask2 = sample['labels'][0,0].clone().numpy(), \
                       sample['labels'][1,0].clone().numpy()
        flow1 = sample['fwdflows'][0].permute(1,2,0).clone().numpy()

        # Project poses onto the RGB images
        poses = sample['poses']
        for j in range(poses.size(1)):
            draw_3d_frame(rgb1, poses[0,j],
                          camera_intrinsics=train_dataset.datasets[0]['camera_intrinsics'])
            draw_3d_frame(rgb2, poses[1,j],
                          camera_intrinsics=train_dataset.datasets[0]['camera_intrinsics'])

        # Show rgb, depth, masks
        plt.figure(100)
        fig.suptitle("Image: {}/{}".format(k, nimgs))
        plt.subplot(331)
        plt.imshow(rgb1)
        plt.title("RGB-1")
        plt.subplot(332)
        plt.imshow(rgb2)
        plt.title("RGB-2")
        plt.subplot(334)
        plt.imshow(normalize_img(depth1, 0.0, 3.0))
        plt.title("Depth-1")
        plt.subplot(335)
        plt.imshow(normalize_img(depth2, 0.0, 3.0))
        plt.title("Depth-2")
        plt.subplot(336)
        plt.imshow(normalize_img(flow1, -0.03, 0.03))
        plt.title("Flow-1-2")
        plt.subplot(337)
        plt.imshow(mask1)
        plt.title("Mask-1")
        plt.subplot(338)
        plt.imshow(mask2)
        plt.title("Mask-2")
        plt.draw()
        plt.pause(0.02)

        # Clear occasionally
        if k % 5 == 0:
            plt.clf()
