import csv
import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import se3layers as se3nn
from torch.autograd import Variable

# NOTE: This is slightly ugly, use this only for the NTfm3D implementation (for use in dataloader)
from layers._ext import se3layers

############
### Helper functions for reading baxter data

# Read baxter state files
def read_baxter_state_file(filename):
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        ret['actjtpos']     = torch.Tensor(spamreader.next()[0:-1])  # Last element is a string due to the way the file is created
        ret['actjtvel']     = torch.Tensor(spamreader.next()[0:-1])
        ret['actjteff']     = torch.Tensor(spamreader.next()[0:-1])
        ret['comjtpos']     = torch.Tensor(spamreader.next()[0:-1])
        ret['comjtvel']     = torch.Tensor(spamreader.next()[0:-1])
        ret['comjtacc']     = torch.Tensor(spamreader.next()[0:-1])
        ret['tarendeffpos'] = torch.Tensor(spamreader.next()[0:-1])
    return ret


# Read baxter SE3 state file for all the joints
def read_baxter_se3state_file(filename):
    # Read all the lines in the SE3-state file
    lines = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in spamreader:
            if len(row) == 0:
                continue
            if type(row[-1]) == str:  # In case we have a string at the end of the list
                row = row[0:-1]
            lines.append(torch.Tensor(row))
    # Parse the SE3-states
    ret, ctr = {}, 0
    while (ctr < len(lines)):
        id = int(lines[ctr][0])  # ID of mesh
        data = lines[ctr + 1].view(3, 4)  # Transform data
        T = torch.eye(4)
        T[0:3, 0:4] = torch.cat([data[0:3, 1:4], data[0:3, 0]], 1)  # [R | t; 0 | 1]
        ret[id] = T  # Add to list of transforms
        ctr += 2  # Increment counter
    return ret


# Read baxter joint labels and their corresponding mesh index value
def read_baxter_labels_file(filename):
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader      = csv.reader(csvfile, delimiter=' ')
        ret['frames']   = spamreader.next()[0:-1]
        ret['meshIds']  = torch.IntTensor([int(x) for x in spamreader.next()[0:-1]])

    return ret


# Read baxter camera data file
def read_cameradata_file(filename):
    # Read lines in the file
    lines = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            lines.append([x for x in row if x != ''])
    # Compute modelview and camera parameter matrix
    ret = {}
    ret['modelView'] = torch.Tensor([float(x) for x in lines[1] + lines[2] + lines[3] + lines[4]]).view(4, 4).clone()
    ret['camParam']  = torch.Tensor([float(x) for x in lines[6] + lines[7] + lines[8] + lines[9]]).view(4, 4).clone()
    return ret


############
### Helper functions for reading image data

# Read depth image from disk
def read_depth_image(filename, ht=240, wd=320, scale=1e-4):
    imgf = cv2.imread(filename, -1).astype(
        np.int16) * scale  # Read image (unsigned short), convert to short & scale to get float
    if (imgf.shape[0] != int(ht) or imgf.shape[1] != int(wd)):
        imgscale = cv2.resize(imgf, (int(wd), int(ht)),
                              interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgf
    return torch.Tensor(imgscale).unsqueeze(0)  # Add extra dimension


# Read flow image from disk
def read_flow_image_xyz(filename, ht=240, wd=320, scale=1e-4):
    imgf = cv2.imread(filename, -1).astype(
        np.int16) * scale  # Read image (unsigned short), convert to short & scale to get float
    if (imgf.shape[0] != int(ht) or imgf.shape[1] != int(wd)):
        imgscale = cv2.resize(imgf, (int(wd), int(ht)),
                              interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgf
    return torch.Tensor(imgscale.transpose((2, 0, 1)))  # NOTE: OpenCV reads BGR so it's already xyz when it is read

#############
### Helper functions for perspective projection stuff
### Computes the pixel x&y grid based on the camera intrinsics assuming perspective projection
def compute_camera_xygrid_from_intrinsics(height, width, intrinsics):
    assert (height > 1 and width > 1)
    assert (intrinsics['fx'] > 0  and intrinsics['fy'] > 0     and
            intrinsics['cx'] >= 0 and intrinsics['cx'] < width and
            intrinsics['cy'] >= 0 and intrinsics['cy'] < height)
    xygrid = torch.ones(1, 2, height, width) # (x,y,1)
    for j in xrange(0, width):  # +x is increasing columns
        xygrid[0, 0, :, j].fill_((j - intrinsics['cx']) / intrinsics['fx'])
    for i in xrange(0, height):  # +y is increasing rows
        xygrid[0, 1, i, :].fill_((i - intrinsics['cy']) / intrinsics['fy'])
    return xygrid

#############
### Helper functions - R/t functions for operating on tensors, not vars

###
### Invert a 3x4 transform (R/t)
def RtInverse(input):
    # Check dimensions
    _, _, nrows, ncols = input.size()
    assert (nrows == 3 and ncols == 4)

    # Init for FWD pass
    input_v = input.view(-1, 3, 4)
    r, t = input_v.narrow(2, 0, 3), input_v.narrow(2, 3, 1)

    # Compute output: [R^T -R^T * t]
    r_o = r.transpose(1, 2)
    t_o = torch.bmm(r_o, t).mul_(-1)
    return torch.cat([r_o, t_o], 2).view_as(input).contiguous()

###
### Compose two tranforms: [R1 t1] * [R2 t2]
def ComposeRtPair(A, B):
    # Check dimensions
    _, _, num_rows, num_cols = A.size()
    assert (num_rows == 3 and num_cols == 4)
    assert (A.is_same_size(B))

    # Init for FWD pass
    Av = A.view(-1, 3, 4)
    Bv = B.view(-1, 3, 4)
    rA, rB = Av.narrow(2, 0, 3), Bv.narrow(2, 0, 3)
    tA, tB = Av.narrow(2, 3, 1), Bv.narrow(2, 3, 1)

    # Compute output
    r = torch.bmm(rA, rB)
    t = torch.baddbmm(tA, rA, tB)
    return torch.cat([r, t], 2).view_as(A).contiguous()

###
### Non-Rigid Transform of 3D points given masks & corrseponding [R t] transforms
def NTfm3D(points, masks, transforms, output=None):
    # Check dimensions
    batch_size, num_channels, data_height, data_width = points.size()
    num_se3 = masks.size()[1]
    assert (num_channels == 3);
    assert (masks.size() == torch.Size([batch_size, num_se3, data_height, data_width]));
    assert (transforms.size() == torch.Size([batch_size, num_se3, 3, 4]));  # Transforms [R|t]
    if output is not None:
        assert(output.is_same_size(points))
    else:
        output = points.clone().zero_()

    # Call the appropriate function to compute the output
    if points.is_cuda:
        se3layers.NTfm3D_forward_cuda(points, masks, transforms, output)
    elif points.type() == 'torch.DoubleTensor':
        se3layers.NTfm3D_forward_double(points, masks, transforms, output)
    else:
        se3layers.NTfm3D_forward_float(points, masks, transforms, output)

    # Return
    return output

### Compute fwd/bwd visibility (for points in time t, which are visible in time t+1 & vice-versa)
# Expects 4D inputs: seq x ndim x ht x wd (or seq x ndim x 3 x 4)
def ComputeFlowAndVisibility(cloud_1, cloud_2, label_1, label_2,
                             poses_1, poses_2, intrinsics,
                             dathreshold=0.01, dawinsize=5):
    # Create memory
    seq, dim, ht, wd = cloud_1.size()
    fwdflows      = torch.FloatTensor(seq, 3, ht, wd).type_as(cloud_1)
    bwdflows      = torch.FloatTensor(seq, 3, ht, wd).type_as(cloud_2)
    fwdvisibility = torch.ByteTensor(seq, 1, ht, wd).cuda() if cloud_1.is_cuda else torch.ByteTensor(seq, 1, ht, wd)
    bwdvisibility = torch.ByteTensor(seq, 1, ht, wd).cuda() if cloud_2.is_cuda else torch.ByteTensor(seq, 1, ht, wd)

    # Compute inverse of poses
    poseinvs_1 = RtInverse(poses_1.clone())
    poseinvs_2 = RtInverse(poses_2.clone())

    # Call cpp/CUDA functions
    if cloud_1.is_cuda:
        assert NotImplementedError, "Only Float version implemented!"
        se3layers.ComputeFlowAndVisibility_cuda(cloud_1,
                                                cloud_2,
                                                label_1,
                                                label_2,
                                                poses_1,
                                                poses_2,
                                                poseinvs_1,
                                                poseinvs_2,
                                                fwdflows,
                                                bwdflows,
                                                fwdvisibility,
                                                bwdvisibility,
                                                intrinsics['fx'],
                                                intrinsics['fy'],
                                                intrinsics['cx'],
                                                intrinsics['cy'],
                                                dathreshold,
                                                dawinsize)
    else:
        assert(cloud_1.type() == 'torch.FloatTensor')
        se3layers.ComputeFlowAndVisibility_float(cloud_1,
                                                 cloud_2,
                                                 label_1,
                                                 label_2,
                                                 poses_1,
                                                 poses_2,
                                                 poseinvs_1,
                                                 poseinvs_2,
                                                 fwdflows,
                                                 bwdflows,
                                                 fwdvisibility,
                                                 bwdvisibility,
                                                 intrinsics['fx'],
                                                 intrinsics['fy'],
                                                 intrinsics['cx'],
                                                 intrinsics['cy'],
                                                 dathreshold,
                                                 dawinsize)

    # Return
    return fwdflows, bwdflows, fwdvisibility, bwdvisibility

############
###  SETUP DATASETS: RECURRENT VERSIONS FOR BAXTER DATA - FROM NATHAN'S BAG FILE

### Helper functions for reading the data directories & loading train/test files
def read_recurrent_baxter_dataset(load_dirs, img_suffix, step_len, seq_len, train_per=0.6, val_per=0.15):
    # Get all the load directories
    if type(load_dirs) == str: # BWDs compatibility
        load_dirs = load_dirs.split(',,')  # Get all the load directories
    assert (train_per + val_per < 1);  # Train + val < test

    # Iterate over each load directory to find the datasets
    datasets = []
    for load_dir in load_dirs:
        if os.path.exists(load_dir + '/postprocessstats.txt'): # This dataset is made up of multiple sub motions
            # Load stats file, get num images & sub-dirs
            statsfilename = load_dir + '/postprocessstats.txt'
            max_flow_step = (step_len * seq_len) # This is the maximum future step (k) for which we need flows (flow_k/)
            with open(statsfilename, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                dirnames, numdata, numinvalid = [], [], 0
                for row in reader:
                    invalid = int(row[0]) # If this = 1, the data for this row is not valid!
                    if invalid:
                        numinvalid += 1
                    else:
                        nimages = int(row[3])
                        numdata.append(int(nimages - max_flow_step)) # We only have flows for these many images!
                        dirnames.append(row[5])
                print('Found {} motions ({} images, {} invalid) in dataset: {}'.format(len(numdata), sum(numdata), numinvalid, load_dir))

            # Setup training and test splits in the dataset, here we actually split based on the sub-dirs
            ndirs = len(dirnames)
            ndirtrain, ndirval = int(train_per * ndirs), int(val_per * ndirs) # First train_per datasets for training, next val_per for validation
            ndirtest = int(ndirs - (ndirtrain + ndirval)) # Rest dirs are for testing

            # Get number of images in the datasets
            nimages = sum(numdata)
            ntrain = sum(numdata[:ndirtrain]) # Num images for training
            nval   = sum(numdata[ndirtrain:ndirtrain+ndirval]) # Validation
            ntest  = nimages - (ntrain + nval) # Number of test images
            print('\tNum train: {} ({}), val: {} ({}), test: {} ({})'.format(
                ndirtrain, ntrain, ndirval, nval, ndirtest, ntest))

            # Setup the dataset structure
            numdata.insert(0, 0) # Add a zero in front for the cumsum
            dataset = {'path'   : load_dir,
                       'suffix' : img_suffix,
                       'step'   : step_len,
                       'seq'    : seq_len,
                       'numdata': nimages,
                       'train'  : [0, ntrain - 1],
                       'val'    : [ntrain, ntrain + nval - 1],
                       'test'   : [ntrain + nval, nimages - 1],
                       'subdirs': {'dirnames': dirnames,
                                   'datahist': np.cumsum(numdata),
                                   'train'   : [0, ndirtrain - 1],
                                   'val'     : [ndirtrain, ndirtrain + ndirval - 1],
                                   'test'    : [ndirtrain + ndirval, ndirs - 1]}
                       }  # start & end inclusive
            datasets.append(dataset)
        else:
            # Get folder names & data statistics for a single load-directory
            dirs = os.listdir(load_dir)
            for dir in dirs:
                path = os.path.join(load_dir, dir) + '/'
                if (os.path.isdir(path)):
                    # Get number of images in the folder
                    statsfilename = os.path.join(path, 'postprocessstats.txt')
                    assert (os.path.exists(statsfilename))
                    max_flow_step = int(step_len * seq_len)  # This is the maximum future step (k) for which we need flows (flow_k/)
                    with open(statsfilename, 'rb') as csvfile:
                        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                        nimages = int(reader.next()[0]) - max_flow_step # We only have flows for these many images!
                        print('Found {} images in the dataset: {}'.format(int(nimages), path))

                    # Setup training and test splits in the dataset
                    ntrain = int(train_per * nimages)  # Use first train_per images for training
                    nval = int(val_per * nimages)  # Use next val_per images for validation set
                    ntest = int(nimages - (ntrain + nval))  # Use remaining images as test set

                    dataset = {'path'   : path,
                               'suffix' : img_suffix,
                               'step'   : step_len,
                               'seq'    : seq_len,
                               'numdata': nimages,
                               'train'  : [0, ntrain - 1],
                               'val'    : [ntrain, ntrain + nval - 1],
                               'test'   : [ntrain + nval, nimages - 1]}  # start & end inclusive
                    datasets.append(dataset)

    return datasets

### Generate the data files (with all the depth, flow etc.) for each sequence
def generate_baxter_sequence(dataset, idx):
    # Get stuff from the dataset
    step, seq, suffix = dataset['step'], dataset['seq'], dataset['suffix']
    # If the dataset has subdirs, find the proper sub-directory to use
    if (dataset.has_key('subdirs')):
        # Find the sub-directory the data falls into
        assert (idx < dataset['numdata']);  # Check if we are within limits
        did = np.searchsorted(dataset['subdirs']['datahist'], idx, 'right') - 1  # ID of sub-directory. If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID
        # Update the ID and path so that we get the correct images
        id   = idx - dataset['subdirs']['datahist'][did] # ID of image within the sub-directory
        path = dataset['path'] + '/' + dataset['subdirs']['dirnames'][did] + '/' # Get the path of the sub-directory
    else:
        id   = idx # Use passed in ID directly
        path = dataset['path'] # Root of dataset
    # Setup start/end IDs of the sequence
    start, end = id, id + (step * seq)
    sequence, ct, stepid = {}, 0, step
    for k in xrange(start, end + 1, step):
        sequence[ct] = {'depth'     : path + 'depth' + suffix + str(k) + '.png',
                        'label'     : path + 'labels' + suffix + str(k) + '.png',
                        'state1'    : path + 'state' + str(k) + '.txt',
                        'state2'    : path + 'state' + str(k + 1) + '.txt',
                        'se3state1' : path + 'se3state' + str(k) + '.txt',
                        'se3state2' : path + 'se3state' + str(k + 1) + '.txt',
                        'flow'   : path + 'flow_' + str(stepid) + '/flow' + suffix + str(start) + '.png',
                        'visible': path + 'flow_' + str(stepid) + '/visible' + suffix + str(start) + '.png'}
        stepid += step  # Get flow from start image to the next step
        ct += 1  # Increment counter
    return sequence

############
### DATA LOADERS: FUNCTION TO LOAD DATA FROM DISK & TORCH DATASET CLASS

### Load baxter sequence from disk
def read_baxter_sequence_from_disk(dataset, id, img_ht=240, img_wd=320, img_scale=1e-4,
                                   ctrl_type='actdiffvel', mesh_ids=torch.Tensor(),
                                   camera_extrinsics={}, camera_intrinsics={},
                                   compute_bwdflows=True):
    # Setup vars
    num_ctrl = 14 if ctrl_type.find('both') else 7      # Num ctrl dimensions
    num_meshes = mesh_ids.nelement()  # Num meshes
    seq_len, step_len = dataset['seq'], dataset['step'] # Get sequence & step length

    # Setup memory
    sequence = generate_baxter_sequence(dataset, id)  # Get the file paths
    points     = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd)
    actconfigs = torch.FloatTensor(seq_len + 1, 7)
    comconfigs = torch.FloatTensor(seq_len + 1, 7)
    controls   = torch.FloatTensor(seq_len, num_ctrl)
    poses      = torch.FloatTensor(seq_len + 1, mesh_ids.nelement() + 1, 3, 4).zero_()

    # For computing FWD/BWD visibilities and FWD/BWD flows
    allposes   = torch.FloatTensor()
    labels     = torch.ByteTensor(seq_len + 1, 1, img_ht, img_wd)  # intially save labels in channel 0 of masks

    # Setup temp var for depth
    depths = points.narrow(1,2,1)  # Last channel in points is the depth

    # Setup vars for BWD flow computation
    if compute_bwdflows:
        masks = torch.ByteTensor( seq_len + 1, num_meshes+1, img_ht, img_wd)

    # Load sequence
    dt = step_len * (1.0 / 30.0)
    for k in xrange(len(sequence)):
        # Get data table
        s = sequence[k]

        # Load depth
        depths[k] = read_depth_image(s['depth'], img_ht, img_wd, img_scale) # Third channel is depth (x,y,z)

        # Load label
        labels[k] = torch.ByteTensor(cv2.imread(s['label'], -1)) # Put the masks in the first channel

        # Load configs
        state = read_baxter_state_file(s['state1'])
        actconfigs[k] = state['actjtpos']
        comconfigs[k] = state['comjtpos']

        # Load SE3 state & get all poses
        se3state = read_baxter_se3state_file(s['se3state1'])
        if allposes.nelement() == 0:
            allposes.resize_(seq_len + 1, len(se3state)+1, 3, 4).fill_(0) # Setup size
        allposes[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        for id, tfm in se3state.iteritems():
            se3tfm = torch.mm(camera_extrinsics['modelView'], tfm)  # NOTE: Do matrix multiply, not * (cmul) here. Camera data is part of options
            allposes[k][id] = se3tfm[0:3, :] # 3 x 4 transform (id is 1-indexed already, 0 is BG)

        # Get poses of meshes we are moving
        poses[k,0,:,0:3] = torch.eye(3).float()  # Identity transform for BG
        for j in xrange(num_meshes):
            meshid = mesh_ids[j]
            poses[k][j+1] = allposes[k][meshid][0:3,:]  # 3 x 4 transform

        # Load controls and FWD flows (for the first "N" items)
        if k < seq_len:
            # Load controls
            if ctrl_type == 'comvel':  # Right arm joint velocities
                controls[k] = state['comjtvel']
            elif ctrl_type == 'actvel':
                controls[k] = state['actjtvel']
            elif ctrl_type == 'comacc':  # Right arm joint accelerations
                controls[k] = state['comjtacc']
            elif ctrl_type == 'comboth':
                controls[k][0:7] = state['comjtvel']  # 0-6  = Joint velocities
                controls[k][7:14] = state['comjtacc']  # 7-13 = Joint accelerations

    # Different control types
    if ctrl_type == 'actdiffvel':
        controls = (actconfigs[1:seq_len + 1, :] - actconfigs[0:seq_len, :]) / dt
    elif ctrl_type == 'comdiffvel':
        controls = (comconfigs[1:seq_len + 1, :] - comconfigs[0:seq_len, :]) / dt

    # Compute x & y values for the 3D points (= xygrid * depths)
    xy = points[:,0:2]
    xy.copy_(camera_intrinsics['xygrid'].expand_as(xy)) # = xygrid
    xy.mul_(depths.expand(seq_len + 1, 2, img_ht, img_wd)) # = xygrid * depths

    # Compute masks
    if compute_bwdflows:
        # Compute masks based on the labels and mesh ids (BG is channel 0, and so on)
        # Note, we have saved labels in channel 0 of masks, so we update all other channels first & channel 0 (BG) last
        for j in xrange(num_meshes):
            masks[:, j+1] = labels.eq(mesh_ids[j])  # Mask out that mesh ID
            if (j == num_meshes - 1):
                masks[:, j+1] = labels.ge(mesh_ids[j])  # Everything in the end-effector
        masks[:,0] = masks.narrow(1,1,num_meshes).sum(1).eq(0)  # All other masks are BG

    # Compute the flows and visibility
    tarpts    = points[1:]    # t+1, t+2, t+3, ....
    initpt    = points[0:1].expand_as(tarpts)
    tarlabels = labels[1:]    # t+1, t+2, t+3, ....
    initlabel = labels[0:1].expand_as(tarlabels)
    tarposes  = allposes[1:]  # t+1, t+2, t+3, ....
    initpose  = allposes[0:1].expand_as(tarposes)

    # Compute flow and visibility
    fwdflows, bwdflows, \
    fwdvisibilities, bwdvisibilities = ComputeFlowAndVisibility(initpt, tarpts, initlabel, tarlabels,
                                                                initpose, tarposes, camera_intrinsics)

    # Return loaded data
    data = {'points': points, 'fwdflows': fwdflows, 'fwdvisibilities': fwdvisibilities,
            'controls': controls, 'actconfigs': actconfigs, 'comconfigs': comconfigs, 'poses': poses}
    if compute_bwdflows:
        data['masks']           = masks
        data['bwdflows']        = bwdflows
        data['bwdvisibilities'] = bwdvisibilities

    return data

### Dataset for Baxter Sequences
class BaxterSeqDataset(Dataset):
    ''' Datasets for training SE3-Nets based on Baxter Sequential data '''

    def __init__(self, datasets, load_function, dtype='train'):
        '''
        Create the data loader given paths to existing list of datasets:
        :param datasets: 		List of datasets that have train | test | val splits
        :param load_function:	Function for reading data from disk given a dataset and an ID (this function needs to
                                return a dictionary of torch tensors)
        :param dtype:			Type of dataset: 'train', 'test' or 'val'
        '''
        assert (len(datasets) > 0);  # Need atleast one dataset
        assert (dtype == 'train' or dtype == 'val' or dtype == 'test')  # Has to be one of the types
        self.datasets = datasets
        self.load_function = load_function
        self.dtype = dtype

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
        sample['id'] = idx # Add the ID of the sample in

        # Return
        return sample

    ### Collate the batch together
    def collate_batch(self, batch):
        # Check if there are any nans in the sampled poses. If there are, then discard the sample
        filtered_batch = []
        for sample in batch:
            if sample['poses'].eq(sample['poses']).all():  # test for nans
                filtered_batch.append(sample)
                # else:
                #    print('Found a dataset with NaNs in the poses. Discarding it')

        # Collate the other samples together using the default collate function
        collated_batch = torch.utils.data.dataloader.default_collate(filtered_batch)

        # Return post-processed batch
        return collated_batch
