# General imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import sys
import cv2

# Torch imports
import torch

opengl = False

def RGBToDepth(img):
    return img[:,:,0]+.01*img[:,:,1]+.0001*img[:,:,2]

def RGBAToMask(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        #print(i, dim, shift, buf[0,0,dim], np.left_shift(buf[0,0,dim], shift))
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

def ConvertPNGListToNumpy(data):
    length = len(data)
    imgs = []
    for raw in data:
        imgs.append(PNGToNumpy(raw))
    arr = np.array(imgs)
    return arr

def ConvertDepthPNGListToNumpy(data):
    length = len(data)
    imgs = []
    for raw in data:
        imgs.append(RGBToDepth(PNGToNumpy(raw)))
    arr = np.array(imgs)
    return arr


############
### Helper functions for reading baxter data

try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

### Load baxter sequence from disk
def read_block_sequence_from_disk(dataset, id, img_ht=240, img_wd=320, img_scale=1e-4,
                                   ctrl_type='actdiffvel', num_ctrl=7,
                                   # num_state=7,
                                   mesh_ids=torch.Tensor(),
                                   # ctrl_ids=torch.LongTensor(),
                                   # camera_extrinsics={}, camera_intrinsics=[],
                                   compute_bwdflows=True, load_color=None, num_tracker=0,
                                   dathreshold=0.01, dawinsize=5, use_only_da=False,
                                   noise_func=None, compute_normals=False, maxdepthdiff=0.05,
                                   bismooth_depths=False, bismooth_width=9, bismooth_std=0.001,
                                   compute_bwdnormals=False, supervised_seg_loss=False):
    # Setup vars
    num_meshes = mesh_ids.nelement()  # Num meshes
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics, camera_extrinsics, ctrl_ids = dataset['camintrinsics'], dataset['camextrinsics'], dataset[
        'ctrlids']

    # Setup memory
    sequence, path, folid = generate_baxter_sequence(dataset, id)  # Get the file paths
    points = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd)
    # actconfigs = torch.FloatTensor(seq_len + 1, num_state) # Actual data is same as state dimension
    actctrlconfigs = torch.FloatTensor(seq_len + 1, num_ctrl)  # Ids in actual data belonging to commanded data
    comconfigs = torch.FloatTensor(seq_len + 1, num_ctrl)  # Commanded data is same as control dimension
    controls = torch.FloatTensor(seq_len, num_ctrl)  # Commanded data is same as control dimension
    poses = torch.FloatTensor(seq_len + 1, mesh_ids.nelement() + 1, 3, 4).zero_()

    # For computing FWD/BWD visibilities and FWD/BWD flows
    allposes = torch.FloatTensor()
    labels = torch.ByteTensor(seq_len + 1, 1, img_ht, img_wd)  # intially save labels in channel 0 of masks

    # Setup temp var for depth
    depths = points.narrow(1, 2, 1)  # Last channel in points is the depth

    # Setup vars for BWD flow computation
    if compute_bwdflows or supervised_seg_loss:
        masks = torch.ByteTensor(seq_len + 1, num_meshes + 1, img_ht, img_wd)

    # Setup vars for color image
    if load_color:
        rgbs = torch.ByteTensor(seq_len + 1, 3, img_ht, img_wd)
        # actctrlvels = torch.FloatTensor(seq_len + 1, num_ctrl)     # Actual data is same as state dimension
        # comvels = torch.FloatTensor(seq_len + 1, num_ctrl)         # Commanded data is same as control dimension

    # Setup vars for tracker data
    if num_tracker > 0:
        trackerconfigs = torch.FloatTensor(seq_len + 1, num_tracker)  # Tracker data is same as tracker dimension

    ## Read camera extrinsics (can be separate per dataset now!)
    try:
        camera_extrinsics = read_cameradata_file(path + '/cameradata.txt')
    except:
        pass  # Can use default cam extrinsics for the entire dataset

    #####
    # Load sequence
    t = torch.linspace(0, seq_len * step_len * (1.0 / 30.0), seq_len + 1).view(seq_len + 1, 1)  # time stamp
    for k in xrange(len(sequence)):
        # Get data table
        s = sequence[k]

        # Load depth
        depths[k] = read_depth_image(s['depth'], img_ht, img_wd, img_scale)  # Third channel is depth (x,y,z)

        # Load label
        # labels[k] = torch.ByteTensor(cv2.imread(s['label'], -1)) # Put the masks in the first channel
        labels[k] = read_label_image(s['label'], img_ht, img_wd)

        # Load configs
        state = read_baxter_state_file(s['state1'])
        # actconfigs[k] = state['actjtpos'] # state dimension
        comconfigs[k] = state['comjtpos']  # ctrl dimension
        actctrlconfigs[k] = state['actjtpos'][ctrl_ids]  # Get states for control IDs
        if state['timestamp'] is not None:
            t[k] = state['timestamp']

        # Load RGB
        if load_color:
            rgbs[k] = read_color_image(s['color'], img_ht, img_wd, colormap=load_color)
            # actctrlvels[k] = state['actjtvel'][ctrl_ids] # Get vels for control IDs
            # comvels[k] = state['comjtvel']

        # Load tracker data
        if num_tracker > 0:
            trackerconfigs[k] = state['trackerjtpos']

        # Load SE3 state & get all poses
        se3state = read_baxter_se3state_file(s['se3state1'])
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

        # Load controls and FWD flows (for the first "N" items)
        if k < seq_len:
            # Load controls
            if ctrl_type == 'comvel':  # Right arm joint velocities
                controls[k] = state['comjtvel']  # ctrl dimension
            elif ctrl_type == 'actvel':
                controls[k] = state['actjtvel'][ctrl_ids]  # state -> ctrl dimension
            elif ctrl_type == 'comacc':  # Right arm joint accelerations
                controls[k] = state['comjtacc']  # ctrl dimension

    # Add noise to the depths before we compute the point cloud
    if (noise_func is not None) and dataset['addnoise']:
        assert (ctrl_type == 'actdiffvel')  # Since we add noise only to the configs
        depths_n = noise_func(depths)
        depths.copy_(depths_n)  # Replace by noisy depths
        # noise_func(depths, actctrlconfigs)

    # Different control types
    dt = t[1:] - t[:-1]  # Get proper dt which can vary between consecutive frames
    if ctrl_type == 'actdiffvel':
        controls = (actctrlconfigs[1:seq_len + 1] - actctrlconfigs[0:seq_len]) / dt  # state -> ctrl dimension
    elif ctrl_type == 'comdiffvel':
        controls = (comconfigs[1:seq_len + 1, :] - comconfigs[0:seq_len, :]) / dt  # ctrl dimension

    # Compute x & y values for the 3D points (= xygrid * depths)
    xy = points[:, 0:2]
    xy.copy_(camera_intrinsics['xygrid'].expand_as(xy))  # = xygrid
    xy.mul_(depths.expand(seq_len + 1, 2, img_ht, img_wd))  # = xygrid * depths

    # Compute masks
    if compute_bwdflows or supervised_seg_loss:
        # Compute masks based on the labels and mesh ids (BG is channel 0, and so on)
        # Note, we have saved labels in channel 0 of masks, so we update all other channels first & channel 0 (BG) last
        for j in xrange(num_meshes):
            masks[:, j + 1] = labels.eq(mesh_ids[j])  # Mask out that mesh ID
            if (j == num_meshes - 1):
                masks[:, j + 1] = labels.ge(mesh_ids[j])  # Everything in the end-effector
        masks[:, 0] = masks.narrow(1, 1, num_meshes).sum(1).eq(0)  # All other masks are BG

    # Compute the flows and visibility
    tarpts = points[1:]  # t+1, t+2, t+3, ....
    initpt = points[0:1].expand_as(tarpts)
    tarlabels = labels[1:]  # t+1, t+2, t+3, ....
    initlabel = labels[0:1].expand_as(tarlabels)
    tarposes = allposes[1:]  # t+1, t+2, t+3, ....
    initpose = allposes[0:1].expand_as(tarposes)

    # Compute flow and visibility
    fwdflows, bwdflows, \
    fwdvisibilities, bwdvisibilities, \
    fwdassocpixelids, bwdassocpixelids = ComputeFlowAndVisibility(initpt, tarpts, initlabel, tarlabels,
                                                                  initpose, tarposes, camera_intrinsics,
                                                                  dathreshold, dawinsize, use_only_da)

    # Compute normal maps & target normal maps (rot/trans of init ones)
    if compute_normals:
        # If asked to do bilateral depth smoothing, do it afresh here
        if bismooth_depths:
            # Compute smoothed depths
            depths_s = BilateralDepthSmoothing(depths, bismooth_width, bismooth_std)
            points_s = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd)  # Create "smoothed" pts
            points_s[:, 2].copy_(depths_s)  # Copy smoothed depths

            # Compute x & y values for the 3D points (= xygrid * depths)
            xy_s = points_s[:, 0:2]
            xy_s.copy_(camera_intrinsics['xygrid'].expand_as(xy_s))  # = xygrid
            xy_s.mul_(depths_s.expand(seq_len + 1, 2, img_ht, img_wd))  # = xygrid * depths

            # Get init and tar pts
            initpt_s = points_s[0:1].expand_as(tarpts)
            tarpts_s = points_s[1:]  # t+1, t+2, t+3, ....
        else:
            initpt_s = initpt  # Use unsmoothed init pts
            tarpts_s = tarpts  # Use unsmoothed tar pts

        tardeltas = ComposeRtPair(tarposes, RtInverse(initpose.clone()))  # Pose_t+1 * Pose_t^-1
        initnormals, tarnormals, \
        validinitnormals, validtarnormals = ComputeNormals(initpt_s, tarpts_s, initlabel, tardeltas,
                                                           maxdepthdiff=maxdepthdiff)

        # Compute normals in the BWD dirn (along with their transformed versions)
        if compute_bwdnormals:
            initdeltas = ComposeRtPair(initpose.clone(), RtInverse(tarposes))  # Pose_t+1 * Pose_t^-1
            bwdinitnormals, bwdtarnormals, \
            validbwdinitnormals, validbwdtarnormals = ComputeNormals(tarpts_s, initpt_s, tarlabels, initdeltas,
                                                                     maxdepthdiff=maxdepthdiff)

    # Return loaded data
    data = {'points': points, 'fwdflows': fwdflows, 'fwdvisibilities': fwdvisibilities, 'folderid': int(folid),
            'fwdassocpixelids': fwdassocpixelids, 'controls': controls, 'comconfigs': comconfigs,
            'poses': poses, 'dt': dt, 'actctrlconfigs': actctrlconfigs}
    if compute_bwdflows:
        data['masks'] = masks
        data['bwdflows'] = bwdflows
        data['bwdvisibilities'] = bwdvisibilities
        data['bwdassocpixelids'] = bwdassocpixelids
    if supervised_seg_loss:
        data['labels'] = masks.max(dim=1)[1]  # Get label image for supervised classification
    if compute_normals:
        data['initnormals'] = initnormals
        data['tarnormals'] = tarnormals
        data['validinitnormals'] = validinitnormals
        data['validtarnormals'] = validtarnormals
        if compute_bwdnormals:
            data['bwdinitnormals'] = bwdinitnormals
            data['bwdtarnormals'] = bwdtarnormals
            data['validbwdinitnormals'] = validbwdinitnormals
            data['validbwdtarnormals'] = validbwdtarnormals
    if load_color:
        data['rgbs'] = rgbs
        # data['labels'] = labels
        # data['actctrlvels'] = actctrlvels
        # data['comvels'] = comvels
    if num_tracker > 0:
        data['trackerconfigs'] = trackerconfigs

    return data


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


###### BOX DATA LOADER
### Load box sequence from disk
def read_box_sequence_from_disk(dataset, id, img_ht=240, img_wd=320, img_scale=1e-4,
                                ctrl_type='ballposforce', num_ctrl=6,
                                compute_bwdflows=True, dathreshold=0.01, dawinsize=5,
                                use_only_da=False, noise_func=None,
                                load_color=False, mesh_ids=torch.Tensor()):  # mesh_ids unused
    # Setup vars
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics = dataset['camintrinsics']

    # Setup memory
    sequence, path = generate_box_sequence(dataset, id)  # Get the file paths
    points = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd)
    states = torch.FloatTensor(seq_len + 1, num_ctrl).zero_()  # All zeros currently
    controls = torch.FloatTensor(seq_len, num_ctrl).zero_()  # Commanded data is same as control dimension
    poses = torch.FloatTensor(seq_len + 1, 3, 3, 4).zero_()

    # For computing FWD/BWD visibilities and FWD/BWD flows
    rgbs = torch.ByteTensor(seq_len + 1, 3, img_ht, img_wd)  # rgbs
    labels = torch.ByteTensor(seq_len + 1, 1, img_ht, img_wd).zero_()  # labels (BG = 0)

    # Setup temp var for depth
    depths = points.narrow(1, 2, 1)  # Last channel in points is the depth

    # Setup vars for BWD flow computation
    if compute_bwdflows:
        masks = torch.ByteTensor(seq_len + 1, 3, img_ht, img_wd)  # BG | Ball | Box

    #####
    # Load sequence
    t = torch.linspace(0, seq_len * step_len * (1.0 / 30.0), seq_len + 1).view(seq_len + 1, 1)  # time stamp
    for k in xrange(len(sequence)):
        # Get data table
        s = sequence[k]

        # Load depth
        depths[k] = read_depth_image(s['depth'], img_ht, img_wd, img_scale)  # Third channel is depth (x,y,z)

        # Load force file
        forcedata = read_forcedata_file(s['force'])
        tarobj = forcedata['targetObject']
        force = forcedata['axis'] * forcedata['magnitude']  # Axis * magnitude

        # Load objectdata file
        objects = read_objectdata_file(s['objects'])
        ballcolor, boxcolor = objects['bullet']['color'], objects[forcedata['targetObject'].split("::")[0]]['color']

        # Load state file
        state = read_box_state_file(s['state'])
        if k < controls.size(0):  # 1 less control than states
            if ctrl_type == 'ballposforce':
                controls[k] = torch.cat([state['bullet::link']['pose'][0:3], force])  # 6D
            elif ctrl_type == 'ballposeforce':
                controls[k] = torch.cat([state['bullet::link']['pose'], force])  # 10D
            elif ctrl_type == 'ballposvelforce':
                controls[k] = torch.cat(
                    [state['bullet::link']['pose'][0:3], state['bullet::link']['vel'][0:3], force])  # 9D
            elif ctrl_type == 'ballposevelforce':
                controls[k] = torch.cat(
                    [state['bullet::link']['pose'], state['bullet::link']['vel'][0:3], force])  # 13D
            else:
                assert False, "Unknown control type: {}".format(ctrl_type)

        # Compute poses (BG | Ball | Box)
        poses[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        campose_w, ballpose_w, boxpose_w = u3d.se3quat_to_rt(
            state['kinect::kinect_camera_depth_optical_frame']['pose']), \
                                           u3d.se3quat_to_rt(state['bullet::link']['pose']), \
                                           u3d.se3quat_to_rt(state[tarobj]['pose'])
        ballpose_c = ComposeRtPair(RtInverse(campose_w[:, 0:3, :].unsqueeze(0)),
                                   ballpose_w[:, 0:3, :].unsqueeze(0))  # Ball in cam = World in cam * Ball in world
        boxpose_c = ComposeRtPair(RtInverse(campose_w[:, 0:3, :].unsqueeze(0)),
                                  boxpose_w[:, 0:3, :].unsqueeze(0))  # Box in cam  = World in cam * Box in world
        poses[k, 1] = ballpose_c;
        poses[k, 1, 0:3, 0:3] = torch.eye(3)  # Ball has identity pose (no rotation for the ball itself)
        poses[k, 2] = boxpose_c  # Box orientation does change

        # Load rgbs & compute labels (0 = BG, 1 = Ball, 2 = Box)
        # NOTE: RGB is loaded BGR so when comparing colors we need to handle it properly
        rgbs[k] = read_color_image(s['color'], img_ht, img_wd)
        ballpix = (((rgbs[k][0] == ballcolor[2]) + (rgbs[k][1] == ballcolor[1]) + (
        rgbs[k][2] == ballcolor[0])) == 3)  # Ball pixels
        boxpix = (
        ((rgbs[k][0] == boxcolor[2]) + (rgbs[k][1] == boxcolor[1]) + (rgbs[k][2] == boxcolor[0])) == 3)  # Box pixels
        labels[k][ballpix], labels[k][boxpix] = 1, 2  # Label all pixels of ball as 1, box as 2

    # Add noise to the depths before we compute the point cloud
    if (noise_func is not None) and dataset['addnoise']:
        depths_n = noise_func(depths)
        depths.copy_(depths_n)  # Replace by noisy depths

    # Different control types
    dt = t[1:] - t[:-1]  # Get proper dt which can vary between consecutive frames

    # Compute x & y values for the 3D points (= xygrid * depths)
    xy = points[:, 0:2]
    xy.copy_(camera_intrinsics['xygrid'].expand_as(xy))  # = xygrid
    xy.mul_(depths.expand(seq_len + 1, 2, img_ht, img_wd))  # = xygrid * depths

    # Compute masks
    if compute_bwdflows:
        # Compute masks based on the labels and mesh ids (BG is channel 0, and so on)
        # Note, we have saved labels in channel 0 of masks, so we update all other channels first & channel 0 (BG) last
        for j in xrange(3):
            masks[:, j] = labels.eq(j)  # Mask out that mesh ID

    # Compute the flows and visibility
    tarpts = points[1:]  # t+1, t+2, t+3, ....
    initpt = points[0:1].expand_as(tarpts)
    tarlabels = labels[1:]  # t+1, t+2, t+3, ....
    initlabel = labels[0:1].expand_as(tarlabels)
    tarposes = poses[1:]  # t+1, t+2, t+3, ....
    initpose = poses[0:1].expand_as(tarposes)

    # Compute flow and visibility
    fwdflows, bwdflows, \
    fwdvisibilities, bwdvisibilities, \
    fwdassocpixelids, bwdassocpixelids = ComputeFlowAndVisibility(initpt, tarpts, initlabel, tarlabels,
                                                                  initpose, tarposes, camera_intrinsics,
                                                                  dathreshold, dawinsize, use_only_da)

    # Return loaded data
    data = {'points': points, 'fwdflows': fwdflows, 'fwdvisibilities': fwdvisibilities,
            'fwdassocpixelids': fwdassocpixelids,
            'states': states, 'controls': controls, 'poses': poses, 'dt': dt}
    if compute_bwdflows:
        data['masks'] = masks
        data['bwdflows'] = bwdflows
        data['bwdvisibilities'] = bwdvisibilities
        data['bwdassocpixelids'] = bwdassocpixelids
    if load_color:
        data['rgbs'] = rgbs

    return data

###################### DATASET
### Dataset for Baxter Sequences
class BaxterSeqDataset(Dataset):
    ''' Datasets for training SE3-Nets based on Baxter Sequential data '''

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
        did = np.digitize(idx,
                          self.datahist) - 1  # If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID

        # Find ID of sample in that dataset (not the same as idx as we might have multiple datasets)
        start = self.datasets[did][self.dtype][
            0]  # This is the ID of the starting sample of the train/test/val part in the entire dataset
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
            # Check if there are NaNs in the poses (BWDs compatibility)
            filtered_batch = []
            for sample in batch:
                if sample['poses'].eq(sample['poses']).all():
                    filtered_batch.append(sample)

            ### In case all these samples have NaN poses, the batch is bad!
            if len(filtered_batch) == 0:
                return None

        # Collate the other samples together using the default collate function
        collated_batch = torch.utils.data.dataloader.default_collate(filtered_batch)

        # Return post-processed batch
        return collated_batch
