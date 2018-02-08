import csv
import torch
import numpy as np
import cv2
import os
import math
from torch.utils.data import Dataset
import se3layers as se3nn
from torch.autograd import Variable
import util.util3d as u3d

# NOTE: This is slightly ugly, use this only for the NTfm3D implementation (for use in dataloader)
from layers._ext import se3layers

############
### Helper functions for reading baxter data

try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

# Read baxter state files
def read_baxter_state_file(filename):
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        ret['actjtpos']     = torch.Tensor(next(spamreader)[0:-1])  # Last element is a string due to the way the file is created
        ret['actjtvel']     = torch.Tensor(next(spamreader)[0:-1])
        ret['actjteff']     = torch.Tensor(next(spamreader)[0:-1])
        ret['comjtpos']     = torch.Tensor(next(spamreader)[0:-1])
        ret['comjtvel']     = torch.Tensor(next(spamreader)[0:-1])
        ret['comjtacc']     = torch.Tensor(next(spamreader)[0:-1])
        ret['tarendeffpos'] = torch.Tensor(next(spamreader)[0:-1])
        try:
            trackdata = next(spamreader)[0:-1]
            ret['trackerjtpos'] = torch.Tensor(trackdata if trackdata[-1] != '' else trackdata[:-1])
            ret['timestamp']    = next(spamreader)[0]
        except:
            ret['trackerjtpos'] = None
            ret['timestamp']    = None
    return ret


# Read baxter SE3 state file for all the joints
def read_baxter_se3state_file(filename):
    # Read all the lines in the SE3-state file
    lines = []
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile,  delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
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
        T[0:3, 0:3] = data[0:3, 1:4]; T[0:3, 3] = data[0:3, 0]
        ret[id] = T # Add to list of transforms
        ctr += 2  # Increment counter
    return ret

# Read baxter joint labels and their corresponding mesh index value
def read_statelabels_file(filename):
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader      = csv.reader(csvfile, delimiter=' ')
        ret['frames']   = next(spamreader)[0:-1]
        ret['meshIds']  = torch.IntTensor([int(x) for x in next(spamreader)[0:-1]])
    return ret

# Read baxter camera data file
def read_cameradata_file(filename):
    # Read lines in the file
    lines = []
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            lines.append([x for x in row if x != ''])
    # Compute modelview and camera parameter matrix
    ret = {}
    ret['modelView'] = torch.Tensor([float(x) for x in lines[1] + lines[2] + lines[3] + lines[4]]).view(4, 4).clone()
    ret['camParam']  = torch.Tensor([float(x) for x in lines[6] + lines[7] + lines[8] + lines[9]]).view(4, 4).clone()
    return ret

# Read baxter camera data file
def read_intrinsics_file(filename):
    # Read lines in the file
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        label = next(spamreader)
        data = next(spamreader)
        assert(len(label) == len(data))
        for k in xrange(len(label)):
            ret[label[k]] = float(data[k])
    return ret

# Read baxter camera data file
# TODO: Make this more general, can actually also store the mesh ids here - could be an alternative to statelabels
def read_statectrllabels_file(filename):
    # Read lines in the file
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile,  delimiter=' ')
        statenames = next(spamreader)[:-1]
        ctrlnames  = next(spamreader)[:-1]
        try:
            trackernames = next(spamreader)[:-1]
        except:
            trackernames = []
    return statenames, ctrlnames, trackernames

############
### Helper functions for reading box data

# Read events in each box/ball dataset
def read_events_file(filename):
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            ret[str(row[0])] = int(row[2])
    return ret

# Read force data
def read_forcedata_file(filename):
    ret = {}
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        ret['axis']         = torch.FloatTensor([float(x) for x in next(reader)]) # Force axis
        ret['point']        = torch.FloatTensor([float(x) for x in next(reader)]) # Force application point
        ret['pointFrame']   = next(reader)[0] # Reference frame in which the force point is expressed in
        ret['magnitude']    = float(next(reader)[0])
        ret['timeStep']     = float(next(reader)[0])
        ret['targetObject'] = next(reader)[0]
    return ret

# Count num of objects in objectdata file. Each object is succeeded by a single empty line
def find_num_objs(filename):
    nobjs, nlines = 0, 0
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            nlines += 1
            if len(row) == 0:
                nobjs += 1 # Empty line separates objects
    return nobjs, nlines

# Read object data file
def read_objectdata_file(filename):
    nobjs, nlines = find_num_objs(filename)
    nlinesperobj = nlines/nobjs
    objs = {}
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        while True:
            # Get line till exit
            try:
                line = next(reader)
            except:
                break # Exit
            # Parse object data
            if len(line) != 0:  # Find lines which are not empty (starts an object)
                obj, name = {}, line[0]
                obj['mass']  = float(next(reader)[0])
                obj['shape'] = next(reader)[0]
                if (obj['shape'] == 'BOX') or (obj['shape'] == 'MESH'):
                    obj['box_size'] = torch.FloatTensor([float(x) for x in next(reader)])
                elif (obj['shape'] == 'SPHERE'):
                    obj['radius'] = float(next(reader)[0])
                else:
                    assert False, 'Unknown object shape: {}'.format(obj['shape'])
                obj['initpose'] = torch.FloatTensor([float(x)*255 for x in next(reader)])
                # Handle color (special)
                if (nlinesperobj == 7):
                    obj['color'] = torch.ByteTensor([int(float(x)*255) for x in next(reader)])
                else:
                    # No colors, choose default color (for box set red, bullet is white, rest is black...)
                    if name.find('bullet') != -1:
                        obj['color'] = torch.ByteTensor([255,255,255]) # White
                    elif name.find('box') != -1:
                        obj['color'] = torch.ByteTensor([255,0,0]) # Red
                    else:
                        obj['color'] = torch.zeros(3) # Black
                # Add to list of objs
                objs[name] = obj # Set object
    return objs # Return

# Read the object state file and return a table per object with the following details:
# Name, Pose, Vel, Accel and Wrench for each object
def read_box_state_file(filename):
    states = {}
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        while True:
            # Get line till exit
            try:
                line = next(reader)
            except:
                break # Exit
            # Parse object data
            if len(line) != 0:  # Find lines which are not empty (starts an object)
                state, name = {}, line[0]
                state['pose']   = torch.FloatTensor([float(x) for x in next(reader)])
                state['vel']    = torch.FloatTensor([float(x) for x in next(reader)])
                state['accel']  = torch.FloatTensor([float(x) for x in next(reader)])
                state['wrench'] = torch.FloatTensor([float(x) for x in next(reader)])
                states[name] = state
    return states

############
### Helper functions for reading image data

# Read depth image from disk
def read_depth_image(filename, ht=240, wd=320, scale=1e-4):
    imgf = cv2.imread(filename, -1).astype(np.int16) * scale  # Read image (unsigned short), convert to short & scale to get float
    if (imgf.shape[0] != int(ht) or imgf.shape[1] != int(wd)):
        imgscale = cv2.resize(imgf, (int(wd), int(ht)), interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgf
    return torch.Tensor(imgscale).unsqueeze(0)  # Add extra dimension

# Read flow image from disk
def read_flow_image_xyz(filename, ht=240, wd=320, scale=1e-4):
    imgf = cv2.imread(filename, -1).astype(np.int16) * scale  # Read image (unsigned short), convert to short & scale to get float
    if (imgf.shape[0] != int(ht) or imgf.shape[1] != int(wd)):
        imgscale = cv2.resize(imgf, (int(wd), int(ht)),
                              interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgf
    return torch.Tensor(imgscale.transpose((2, 0, 1)))  # NOTE: OpenCV reads BGR so it's already xyz when it is read

# Read label image from disk
def read_label_image(filename, ht=240, wd=320):
    imgl = cv2.imread(filename, -1) # This can be an image with 1 or 3 channels. If 3 channel image, choose 2nd channel
    if (imgl.ndim == 3 and imgl.shape[2] == 3):
        imgl = imgl[:,:,1] # Get only 2nd channel (real data)
    if (imgl.shape[0] != int(ht) or imgl.shape[1] != int(wd)):
        imgscale = cv2.resize(imgl, (int(wd), int(ht)), interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgl
    return torch.ByteTensor(imgscale).unsqueeze(0)  # Add extra dimension

# Read label image from disk
def read_color_image(filename, ht=240, wd=320, colormap='rgb'):
    imgl = cv2.imread(filename) # This can be an image with 1 or 3 channels. If 3 channel image, choose 2nd channel
    if (imgl.shape[0] != int(ht) or imgl.shape[1] != int(wd)):
        imgscale = cv2.resize(imgl, (int(wd), int(ht)), interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgl
    # Convert colormaps
    if colormap == 'hsv':
        imgscale = cv2.cvtColor(imgscale, cv2.COLOR_RGB2HSV) # Seems like by default images are RGB, not BGR
    elif colormap != 'rgb':
        assert False, "Wrong colormap input: {}".format(colormap)
    return torch.ByteTensor(imgscale.transpose(2,0,1)).unsqueeze(0)  # Add extra dimension

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
                             dathreshold=0.01, dawinsize=5,
                             use_only_da=False):
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
    else:
        assert (cloud_1.type() == 'torch.FloatTensor')
        if use_only_da:
            # This computes flows based only on points @ t1 that are visible and can be associated to a point @ t2
            # So it assumes that we are only given data-associations as opposed to full tracking information
            # This flow will be noisy for real data. For sim data, it should give the same result as the other function
            local_1 = torch.FloatTensor(seq, 3, ht, wd).type_as(cloud_1)
            local_2 = torch.FloatTensor(seq, 3, ht, wd).type_as(cloud_2)
            se3layers.ComputeFlowAndVisibility_Pts_float(cloud_1,
                                                         cloud_2,
                                                         local_1,
                                                         local_2,
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
            # This computes flows based on the internal tracker poses over time, so we can get flow
            # for every input point, not just the ones that are visible @ t2
            # Flow will be smoother for real data
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

### Compute fwd/bwd visibility (for points in time t, which are visible in time t+1 & vice-versa)
# Expects 4D inputs: seq x ndim x ht x wd (or seq x ndim x 3 x 4)
def ComputeNormals(cloud_1, cloud_2, label_1, delta_12,
                   maxdepthdiff=0.05):
    # Create memory
    seq, dim, ht, wd = cloud_1.size()
    normal_1 = torch.FloatTensor(seq, 3, ht, wd).type_as(cloud_1)
    normal_2 = torch.FloatTensor(seq, 3, ht, wd).type_as(cloud_2)

    # Call cpp/CUDA functions
    if cloud_1.is_cuda:
        assert NotImplementedError, "Only Float version implemented!"
    else:
        assert (cloud_1.type() == 'torch.FloatTensor')
        # This computes normals for the initial cloud and transforms
        # these based on the R/t to get transformed normal clouds
        se3layers.ComputeNormals_float(cloud_1,
                                       cloud_2,
                                       label_1,
                                       delta_12,
                                       normal_1,
                                       normal_2,
                                       maxdepthdiff)
        valid_normal_1 = normal_1.abs().sum(1).gt(0).unsqueeze(1)
        valid_normal_2 = normal_2.abs().sum(1).gt(0).unsqueeze(1)

    # Return
    return normal_1, normal_2, valid_normal_1, valid_normal_2


### Compute fwd/bwd visibility (for points in time t, which are visible in time t+1 & vice-versa)
# Expects 4D inputs: seq x ndim x ht x wd (or seq x ndim x 3 x 4)
def BilateralDepthSmoothing(depth, width=9, depthstd=0.001):
    # Create memory
    seq, dim, ht, wd = depth.size()
    assert(dim == 1) # Depth only
    depth_s = torch.FloatTensor(seq, dim, ht, wd).type_as(depth)

    # Create half of the smoothing kernel (in position space)
    # This is symmetric about zero (so we only go from 0->width/2)
    # It is also same for x & y so it's a symmetric 2D Gaussian with independent x and y
    lockernel = torch.zeros(width//2+1) # TODO: Is this correct for even sized kernels?
    lockernelstd = lockernel.nelement()/2. # Std deviation in pixel space (~1/2 of half the kernel size, for a 9x9 kernel it is 2.5 pixels)
    for k in range(lockernel.nelement()):
        lockernel[k] = math.exp(-(k*k) / (2.0*lockernelstd*lockernelstd))

    # Call cpp/CUDA functions
    if depth.is_cuda:
        assert NotImplementedError, "Only Float version implemented!"
    else:
        assert (depth.type() == 'torch.FloatTensor')
        # This computes normals for the initial cloud and transforms
        # these based on the R/t to get transformed normal clouds
        se3layers.BilateralDepthSmoothing_float(depth,
                                                depth_s,
                                                lockernel,
                                                depthstd)

    # Return
    return depth_s

############
###  SETUP DATASETS: RECURRENT VERSIONS FOR BAXTER DATA - FROM NATHAN'S BAG FILE

### Function that filters the data based - mainly for the real data where we need to check dts
### and other related stuff.
def valid_data_filter(path, nexamples, step, seq, state_labels,
                      mean_dt, std_dt,
                      reject_left_motion=False,
                      reject_right_still=False):
    try:
        ## Read the meta-data to get "timestamps"
        meta_data = np.loadtxt(path + '/trackerdata_meta.txt', skiprows=1)
        timestamps = (meta_data[0:nexamples+step*(seq+1), 1] + 1e-9 * meta_data[0:nexamples+step*(seq+1), 2]) - meta_data[0,0] # Convert to seconds
        ## Read all the state files
        if reject_left_motion or reject_right_still:
            nstate = len(state_labels)
            left_ids  = [state_labels.index(x) for x in ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']]
            right_ids = [state_labels.index(x) for x in ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']]
            jtangles = np.zeros((nexamples+step*(seq+1), nstate), dtype=np.float32)
            for k in xrange(nexamples+step*(seq+1)):
                with open(path + 'state' + str(k) + '.txt', 'rt') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                    jtangles[k]  = next(spamreader)[0:-1]
        ## Compute all the valid examples
        validids = []
        for k in xrange(nexamples):
            # Compute dt & check if dts of all steps are within mean +- 2*std
            dt = timestamps[k+step:k+step*(seq+1):step] - timestamps[k:k+step*seq:step] # Step along the timestamps
            tok = np.all(np.abs(dt - mean_dt) < 2*std_dt)
            # Compute max change in joint angles of left arm. Threshold this
            if reject_left_motion:
                dall = np.abs(jtangles[k+step:k+step*(seq+1):step] - jtangles[k:k+step*seq:step]) # Step along the timestamps
                leftok = (dall[:, left_ids].max() < 0.005) # Max change in left arm < 0.005 radians
            else:
                leftok = True
            # Compute max change in joint angles of right arm.
            # Atleast one joint has to have a decent motion in a sequence
            if reject_right_still:
                dall = np.abs(jtangles[k+step:k+step*(seq+1):step] - jtangles[k:k+step*seq:step]) # Step along the timestamps
                nstill = (dall[:, right_ids].max(1) < 0.005).sum()
                rightok = (nstill < seq/2.) # Atleast half the frames need to have motion
                # All frames need to have motion here (very strict)
                # rightok = True
                # for j in xrange(dall.shape[0]):
                #     if dall[j, right_ids].max() < 0.005: # If all joints in this frame have little motion, discard example
                #         rightok = False
                #         break
            else:
                rightok = True
            # If all tests pass, accept example
            if tok and leftok and rightok:
                validids.append(k) # The entire sequence has dts that are within 2 std.devs of the mean dt
    except:
        print("Failed/Did not run validity check. Using all examples in the dataset")
        validids = range(0, nexamples)  # For sim data, no need for this step
    return validids

### Helper functions for reading the data directories & loading train/test files
def read_recurrent_baxter_dataset(load_dirs, img_suffix, step_len, seq_len, train_per=0.6, val_per=0.15,
                                  valid_filter=None, cam_intrinsics=[], cam_extrinsics =[],
                                  ctrl_ids=[], add_noise=[], state_labels=[]):
    # Get all the load directories
    if type(load_dirs) == str: # BWDs compatibility
        load_dirs = load_dirs.split(',,')  # Get all the load directories
    assert (train_per + val_per <= 1);  # Train + val < test

    # Iterate over each load directory to find the datasets
    datasets = []
    for load_dir in load_dirs:
        if os.path.exists(load_dir + '/postprocessstats.txt'): # This dataset is made up of multiple sub motions (box data is by default like that)
            # Load stats file, get num images & sub-dirs
            statsfilename = load_dir + '/postprocessstats.txt'
            max_flow_step = (step_len * seq_len) # This is the maximum future step (k) for which we need flows (flow_k/)
            with open(statsfilename, 'rt') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                dirnames, numdata, numinvalid = [], [], 0
                for row in reader:
                    invalid = int(row[0]) # If this = 1, the data for this row is not valid!
                    nexamples = int(row[3]) - int(max_flow_step) # Num examples
                    if invalid or (nexamples < 1):
                        numinvalid += 1
                    else:
                        numdata.append(nexamples) # We only have flows for these many images!
                        dirnames.append(row[5])
                print('Found {}/{} valid motions ({} examples) in dataset: {}'.format(len(numdata), numinvalid + len(numdata),
                                                                                      sum(numdata), load_dir))

            # Setup training and test splits in the dataset, here we actually split based on the sub-dirs
            ndirs = len(dirnames)
            ndirtrain, ndirval = int(train_per * ndirs), int(val_per * ndirs) # First train_per datasets for training, next val_per for validation
            ndirtest = int(ndirs - (ndirtrain + ndirval)) # Rest dirs are for testing

            # Get number of images in the datasets
            nexamples = sum(numdata)
            ntrain = sum(numdata[:ndirtrain]) # Num images for training
            nval   = sum(numdata[ndirtrain:ndirtrain+ndirval]) # Validation
            ntest  = nexamples - (ntrain + nval) # Number of test images
            print('\tNum train: {} ({}), val: {} ({}), test: {} ({})'.format(
                ndirtrain, ntrain, ndirval, nval, ndirtest, ntest))

            # Setup the dataset structure
            numdata.insert(0, 0) # Add a zero in front for the cumsum
            dataset = {'path'   : load_dir,
                       'suffix' : img_suffix,
                       'step'   : step_len,
                       'seq'    : seq_len,
                       'numdata': nexamples,
                       'train'  : [0, ntrain - 1],
                       'val'    : [ntrain, ntrain + nval - 1],
                       'test'   : [ntrain + nval, nexamples - 1],
                       'subdirs': {'dirnames': dirnames,
                                   'datahist': np.cumsum(numdata),
                                   'train'   : [0, ndirtrain - 1],
                                   'val'     : [ndirtrain, ndirtrain + ndirval - 1],
                                   'test'    : [ndirtrain + ndirval, ndirs - 1]},
                       }
            if len(cam_intrinsics) > 0:
                dataset['camintrinsics'] = cam_intrinsics[len(datasets)]
            if len(cam_extrinsics) > 0:
                dataset['camextrinsics'] = cam_extrinsics[len(datasets)]
            if len(ctrl_ids) > 0:
                dataset['ctrlids']  = ctrl_ids[len(datasets)]
            if len(add_noise) > 0:
                dataset['addnoise'] = add_noise[len(datasets)]
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
                    with open(statsfilename, 'rt') as csvfile:
                        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                        nexamples = int(next(reader)[0]) - max_flow_step # We only have flows for these many images!

                    # This function checks all examples "apriori" to see if they are valid
                    # and returns a set of ids such that the sequence of examples from that id
                    # to id + seq*step are valid
                    if valid_filter is not None:
                        validids = valid_filter(path, nexamples, step_len, seq_len,
                                                state_labels[len(datasets)])
                    else:
                        validids = range(0, nexamples) # Is just the same as ids, all samples are valid
                    nvalid = len(validids)
                    print('Found {}/{} valid examples ({}%) in the dataset: {}'.format(int(nvalid),
                            int(nexamples), nvalid*(100.0/nexamples), path))  # Setup training and test splits in the dataset

                    # Split up train/test/validation
                    ntrain = int(train_per * nvalid)  # Use first train_per valid examples for training
                    nval   = int(val_per * nvalid)  # Use next val_per valid examples for validation set
                    ntest  = int(nvalid - (ntrain + nval))  # Use remaining valid examples as test set

                    # Create the dataset
                    dataset = {'path'   : path,
                               'suffix' : img_suffix,
                               'step'   : step_len,
                               'seq'    : seq_len,
                               'numdata': nvalid,
                               'ids'    : validids,
                               'train'  : [0, ntrain - 1],
                               'val'    : [ntrain, ntrain + nval - 1],
                               'test'   : [ntrain + nval, nvalid - 1],  # start & end inclusive
                               }
                    if len(cam_intrinsics) > 0:
                        dataset['camintrinsics'] = cam_intrinsics[len(datasets)]
                    if len(cam_extrinsics) > 0:
                        dataset['camextrinsics'] = cam_extrinsics[len(datasets)]
                    if len(ctrl_ids) > 0:
                        dataset['ctrlids']  = ctrl_ids[len(datasets)]
                    if len(add_noise) > 0:
                        dataset['addnoise'] = add_noise[len(datasets)]
                    datasets.append(dataset)

    return datasets

### Generate the data files (with all the depth, flow etc.) for each sequence
def generate_baxter_sequence(dataset, idx):
    # Get stuff from the dataset
    step, seq, suffix = dataset['step'], dataset['seq'], dataset['suffix']
    # If the dataset has subdirs, find the proper sub-directory to use
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
    return sequence, path

### Generate the data files (with all the depth, flow etc.) for each sequence
def generate_box_sequence(dataset, idx):
    # Get stuff from the dataset
    step, seq, suffix = dataset['step'], dataset['seq'], dataset['suffix']
    # Find the sub-directory the data falls into
    assert (idx < dataset['numdata']);  # Check if we are within limits
    did = np.searchsorted(dataset['subdirs']['datahist'], idx, 'right') - 1  # ID of sub-directory. If [0, 10, 20] & we get 10, this will be bin 2 (10-20), so we reduce by 1 to get ID
    # Update the ID and path so that we get the correct images
    id   = idx - dataset['subdirs']['datahist'][did] # ID of image within the sub-directory
    path = dataset['path'] + '/' + dataset['subdirs']['dirnames'][did] + '/' # Get the path of the sub-directory
    # Setup start/end IDs of the sequence
    start, end = id, id + (step * seq)
    sequence, ct, stepid = {}, 0, step
    for k in xrange(start, end + 1, step):
        sequence[ct] = {'depth'   : path + 'depth' + suffix + str(k) + '.png',
                        'color'   : path + 'rgb'   + suffix + str(k) + '.png',
                        'state'   : path + 'state' + str(k) + '.txt',
                        'flow'    : path + 'flow_' + str(stepid) + '/flow' + suffix + str(start) + '.png',
                        'force'   : path + 'forcedata.txt',
                        'objects' : path + 'objectdata.txt'}
        stepid += step  # Get flow from start image to the next step
        ct += 1  # Increment counter
    return sequence, path

############
### DATA LOADERS: FUNCTION TO LOAD DATA FROM DISK & TORCH DATASET CLASS
def add_gaussian_noise(depths, configs, std_d=0.02,
                       scale_d=True, std_j=0.02):
    # Add random gaussian noise to the depths
    noise_d = torch.randn(depths.size()).type_as(depths) * std_d # Sample from 0-mean, 1-std distribution & scale by the std
    if scale_d:
        noise_d.mul_((depths/2.5).clamp_(min=0.25, max=1.0)) # Scale the std.deviation essentially based on the depth
    depths.add_(noise_d).clamp_(min=0) # Add the noise to the depths

    # Add control / config noise
    noise_c = torch.randn(configs.size()).mul_(std_j).clamp_(max=2*std_j, min=-2*std_j)
    configs.add_(noise_c)

def add_edge_based_noise(depths, zthresh=0.04, edgeprob=0.35,
                         defprob=0.005, noisestd=0.005):
    depths_n = depths.clone()
    se3layers.AddNoise_float(depths,    # input depth (unchanged)
                             depths_n,  # noisy depth
                             zthresh,   # zthresh
                             edgeprob,  # edgeprob
                             defprob,   # def prob
                             noisestd)  # noise std
    return depths_n

### Load baxter sequence from disk
def read_baxter_sequence_from_disk(dataset, id, img_ht=240, img_wd=320, img_scale=1e-4,
                                   ctrl_type='actdiffvel', num_ctrl=7,
                                   #num_state=7,
                                   mesh_ids=torch.Tensor(),
                                   #ctrl_ids=torch.LongTensor(),
                                   #camera_extrinsics={}, camera_intrinsics=[],
                                   compute_bwdflows=True, load_color=None, num_tracker=0,
                                   dathreshold=0.01, dawinsize=5, use_only_da=False,
                                   noise_func=None, compute_normals=False, maxdepthdiff=0.05,
                                   bismooth_depths=False, bismooth_width=9, bismooth_std=0.001,
                                   compute_bwdnormals=False, supervised_seg_loss=False):
    # Setup vars
    num_meshes = mesh_ids.nelement()  # Num meshes
    seq_len, step_len = dataset['seq'], dataset['step'] # Get sequence & step length
    camera_intrinsics, camera_extrinsics, ctrl_ids = dataset['camintrinsics'], dataset['camextrinsics'], dataset['ctrlids']

    # Setup memory
    sequence, path = generate_baxter_sequence(dataset, id)  # Get the file paths
    points     = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd)
    #actconfigs = torch.FloatTensor(seq_len + 1, num_state) # Actual data is same as state dimension
    actctrlconfigs = torch.FloatTensor(seq_len + 1, num_ctrl) # Ids in actual data belonging to commanded data
    comconfigs = torch.FloatTensor(seq_len + 1, num_ctrl)  # Commanded data is same as control dimension
    controls   = torch.FloatTensor(seq_len, num_ctrl)      # Commanded data is same as control dimension
    poses      = torch.FloatTensor(seq_len + 1, mesh_ids.nelement() + 1, 3, 4).zero_()

    # For computing FWD/BWD visibilities and FWD/BWD flows
    allposes   = torch.FloatTensor()
    labels     = torch.ByteTensor(seq_len + 1, 1, img_ht, img_wd)  # intially save labels in channel 0 of masks

    # Setup temp var for depth
    depths = points.narrow(1,2,1)  # Last channel in points is the depth
    
    # Setup vars for BWD flow computation
    if compute_bwdflows or supervised_seg_loss:
        masks = torch.ByteTensor( seq_len + 1, num_meshes+1, img_ht, img_wd)

    # Setup vars for color image
    if load_color:
        rgbs = torch.ByteTensor(seq_len + 1, 3, img_ht, img_wd)
        #actctrlvels = torch.FloatTensor(seq_len + 1, num_ctrl)     # Actual data is same as state dimension
        #comvels = torch.FloatTensor(seq_len + 1, num_ctrl)         # Commanded data is same as control dimension

    # Setup vars for tracker data
    if num_tracker > 0:
        trackerconfigs = torch.FloatTensor(seq_len + 1, num_tracker)  # Tracker data is same as tracker dimension

    ## Read camera extrinsics (can be separate per dataset now!)
    try:
        camera_extrinsics = read_cameradata_file(path + '/cameradata.txt')
    except:
        pass # Can use default cam extrinsics for the entire dataset

    #####
    # Load sequence
    t = torch.linspace(0, seq_len*step_len*(1.0/30.0), seq_len+1).view(seq_len+1,1) # time stamp
    for k in xrange(len(sequence)):
        # Get data table
        s = sequence[k]

        # Load depth
        depths[k] = read_depth_image(s['depth'], img_ht, img_wd, img_scale) # Third channel is depth (x,y,z)

        # Load label
        #labels[k] = torch.ByteTensor(cv2.imread(s['label'], -1)) # Put the masks in the first channel
        labels[k] = read_label_image(s['label'], img_ht, img_wd)

        # Load configs
        state = read_baxter_state_file(s['state1'])
        #actconfigs[k] = state['actjtpos'] # state dimension
        comconfigs[k] = state['comjtpos'] # ctrl dimension
        actctrlconfigs[k] = state['actjtpos'][ctrl_ids] # Get states for control IDs
        if state['timestamp'] is not None:
            t[k] = state['timestamp']

        # Load RGB
        if load_color:
            rgbs[k] = read_color_image(s['color'], img_ht, img_wd, colormap=load_color)
            #actctrlvels[k] = state['actjtvel'][ctrl_ids] # Get vels for control IDs
            #comvels[k] = state['comjtvel']

        # Load tracker data
        if num_tracker > 0:
            trackerconfigs[k] = state['trackerjtpos']

        # Load SE3 state & get all poses
        se3state = read_baxter_se3state_file(s['se3state1'])
        if allposes.nelement() == 0:
            allposes.resize_(seq_len + 1, len(se3state)+1, 3, 4).fill_(0) # Setup size
        allposes[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        for id, tfm in se3state.items():
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
                controls[k] = state['comjtvel'] # ctrl dimension
            elif ctrl_type == 'actvel':
                controls[k] = state['actjtvel'][ctrl_ids] # state -> ctrl dimension
            elif ctrl_type == 'comacc':  # Right arm joint accelerations
                controls[k] = state['comjtacc'] # ctrl dimension

    # Add noise to the depths before we compute the point cloud
    if (noise_func is not None) and dataset['addnoise']:
        assert(ctrl_type == 'actdiffvel') # Since we add noise only to the configs
        depths_n = noise_func(depths)
        depths.copy_(depths_n) # Replace by noisy depths
        #noise_func(depths, actctrlconfigs)

    # Different control types
    dt = t[1:] - t[:-1] # Get proper dt which can vary between consecutive frames
    if ctrl_type == 'actdiffvel':
        controls = (actctrlconfigs[1:seq_len+1] - actctrlconfigs[0:seq_len]) / dt # state -> ctrl dimension
    elif ctrl_type == 'comdiffvel':
        controls = (comconfigs[1:seq_len+1, :] - comconfigs[0:seq_len, :]) / dt # ctrl dimension

    # Compute x & y values for the 3D points (= xygrid * depths)
    xy = points[:,0:2]
    xy.copy_(camera_intrinsics['xygrid'].expand_as(xy)) # = xygrid
    xy.mul_(depths.expand(seq_len + 1, 2, img_ht, img_wd)) # = xygrid * depths

    # Compute masks
    if compute_bwdflows or supervised_seg_loss:
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
                                                                initpose, tarposes, camera_intrinsics,
                                                                dathreshold, dawinsize, use_only_da)

    # Compute normal maps & target normal maps (rot/trans of init ones)
    if compute_normals:
        # If asked to do bilateral depth smoothing, do it afresh here
        if bismooth_depths:
            # Compute smoothed depths
            depths_s = BilateralDepthSmoothing(depths, bismooth_width, bismooth_std)
            points_s = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd) # Create "smoothed" pts
            points_s[:,2].copy_(depths_s) # Copy smoothed depths

            # Compute x & y values for the 3D points (= xygrid * depths)
            xy_s = points_s[:, 0:2]
            xy_s.copy_(camera_intrinsics['xygrid'].expand_as(xy_s))  # = xygrid
            xy_s.mul_(depths_s.expand(seq_len + 1, 2, img_ht, img_wd))  # = xygrid * depths

            # Get init and tar pts
            initpt_s = points_s[0:1].expand_as(tarpts)
            tarpts_s = points_s[1:]  # t+1, t+2, t+3, ....
        else:
            initpt_s = initpt # Use unsmoothed init pts
            tarpts_s = tarpts # Use unsmoothed tar pts

        tardeltas = ComposeRtPair(tarposes, RtInverse(initpose.clone()))  # Pose_t+1 * Pose_t^-1
        initnormals, tarnormals,\
        validinitnormals, validtarnormals = ComputeNormals(initpt_s, tarpts_s, initlabel, tardeltas,
                                                           maxdepthdiff=maxdepthdiff)

        # Compute normals in the BWD dirn (along with their transformed versions)
        if compute_bwdnormals:
            initdeltas = ComposeRtPair(initpose.clone(), RtInverse(tarposes))  # Pose_t+1 * Pose_t^-1
            bwdinitnormals, bwdtarnormals, \
            validbwdinitnormals, validbwdtarnormals = ComputeNormals(tarpts_s, initpt_s, tarlabels, initdeltas,
                                                                     maxdepthdiff=maxdepthdiff)

    # Return loaded data
    data = {'points': points, 'fwdflows': fwdflows, 'fwdvisibilities': fwdvisibilities,
            'controls': controls, 'comconfigs': comconfigs, 'poses': poses,
            'dt': dt, 'actctrlconfigs': actctrlconfigs}
    if compute_bwdflows:
        data['masks']           = masks
        data['bwdflows']        = bwdflows
        data['bwdvisibilities'] = bwdvisibilities
    if supervised_seg_loss:
        data['labels'] = masks.max(dim=1)[1] # Get label image for supervised classification
    if compute_normals:
        data['initnormals'] = initnormals
        data['tarnormals']  = tarnormals
        data['validinitnormals'] = validinitnormals
        data['validtarnormals']  = validtarnormals
        if compute_bwdnormals:
            data['bwdinitnormals'] = bwdinitnormals
            data['bwdtarnormals']  = bwdtarnormals
            data['validbwdinitnormals'] = validbwdinitnormals
            data['validbwdtarnormals']  = validbwdtarnormals
    if load_color:
        data['rgbs']   = rgbs
        #data['labels'] = labels
        #data['actctrlvels'] = actctrlvels
        #data['comvels'] = comvels
    if num_tracker > 0:
        data['trackerconfigs'] = trackerconfigs

    return data

def filter_func(batch, mean_dt, std_dt):
    # Check if there are any nans in the sampled poses. If there are, then discard the sample
    filtered_batch = []
    for sample in batch:
        # Check if any dt is too large (within 2 std. deviations of the mean)
        tok    = ((sample['dt'] - mean_dt).abs_() < 2*std_dt).all()
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
                                load_color=False, mesh_ids=torch.Tensor()): # mesh_ids unused
    # Setup vars
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics = dataset['camintrinsics']

    # Setup memory
    sequence, path = generate_box_sequence(dataset, id)  # Get the file paths
    points   = torch.FloatTensor(seq_len + 1, 3, img_ht, img_wd)
    states   = torch.FloatTensor(seq_len + 1, num_ctrl).zero_()  # All zeros currently
    controls = torch.FloatTensor(seq_len, num_ctrl).zero_()  # Commanded data is same as control dimension
    poses    = torch.FloatTensor(seq_len + 1, 3, 3, 4).zero_()

    # For computing FWD/BWD visibilities and FWD/BWD flows
    rgbs   = torch.ByteTensor(seq_len + 1, 3, img_ht, img_wd)  # rgbs
    labels = torch.ByteTensor(seq_len + 1, 1, img_ht, img_wd).zero_() # labels (BG = 0)

    # Setup temp var for depth
    depths = points.narrow(1, 2, 1)  # Last channel in points is the depth

    # Setup vars for BWD flow computation
    if compute_bwdflows:
        masks = torch.ByteTensor(seq_len + 1, 3, img_ht, img_wd) # BG | Ball | Box

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
        force = forcedata['axis'] * forcedata['magnitude'] # Axis * magnitude

        # Load objectdata file
        objects = read_objectdata_file(s['objects'])
        ballcolor, boxcolor = objects['bullet']['color'], objects[forcedata['targetObject'].split("::")[0]]['color']

        # Load state file
        state = read_box_state_file(s['state'])
        if k < controls.size(0): # 1 less control than states
            if ctrl_type == 'ballposforce':
                controls[k] = torch.cat([state['bullet::link']['pose'][0:3], force]) # 6D
            elif ctrl_type == 'ballposeforce':
                controls[k] = torch.cat([state['bullet::link']['pose'], force])  # 10D
            elif ctrl_type == 'ballposvelforce':
                controls[k] = torch.cat([state['bullet::link']['pose'][0:3], state['bullet::link']['vel'][0:3], force]) # 9D
            elif ctrl_type == 'ballposevelforce':
                controls[k] = torch.cat([state['bullet::link']['pose'], state['bullet::link']['vel'][0:3], force]) # 13D
            else:
                assert False, "Unknown control type: {}".format(ctrl_type)

        # Compute poses (BG | Ball | Box)
        poses[k,0,:,0:3]  = torch.eye(3).float()  # Identity transform for BG
        campose_w, ballpose_w, boxpose_w = u3d.se3quat_to_rt(state['kinect::kinect_camera_depth_optical_frame']['pose']),\
                                           u3d.se3quat_to_rt(state['bullet::link']['pose']),\
                                           u3d.se3quat_to_rt(state[tarobj]['pose'])
        ballpose_c = ComposeRtPair(RtInverse(campose_w[:,0:3,:].unsqueeze(0)), ballpose_w[:,0:3,:].unsqueeze(0)) # Ball in cam = World in cam * Ball in world
        boxpose_c  = ComposeRtPair(RtInverse(campose_w[:,0:3,:].unsqueeze(0)), boxpose_w[:,0:3,:].unsqueeze(0))  # Box in cam  = World in cam * Box in world
        poses[k,1] = ballpose_c; poses[k,1,0:3,0:3] = torch.eye(3) # Ball has identity pose (no rotation for the ball itself)
        poses[k,2] = boxpose_c # Box orientation does change

        # Load rgbs & compute labels (0 = BG, 1 = Ball, 2 = Box)
        # NOTE: RGB is loaded BGR so when comparing colors we need to handle it properly
        rgbs[k] = read_color_image(s['color'], img_ht, img_wd)
        ballpix = (((rgbs[k][0] == ballcolor[2]) + (rgbs[k][1] == ballcolor[1]) + (rgbs[k][2] == ballcolor[0])) == 3) # Ball pixels
        boxpix  = (((rgbs[k][0] == boxcolor[2]) + (rgbs[k][1] == boxcolor[1]) + (rgbs[k][2] == boxcolor[0])) == 3) # Box pixels
        labels[k][ballpix], labels[k][boxpix] = 1, 2 # Label all pixels of ball as 1, box as 2

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
            masks[:,j] = labels.eq(j)  # Mask out that mesh ID

    # Compute the flows and visibility
    tarpts = points[1:]  # t+1, t+2, t+3, ....
    initpt = points[0:1].expand_as(tarpts)
    tarlabels = labels[1:]  # t+1, t+2, t+3, ....
    initlabel = labels[0:1].expand_as(tarlabels)
    tarposes = poses[1:]  # t+1, t+2, t+3, ....
    initpose = poses[0:1].expand_as(tarposes)

    # Compute flow and visibility
    fwdflows, bwdflows, \
    fwdvisibilities, bwdvisibilities = ComputeFlowAndVisibility(initpt, tarpts, initlabel, tarlabels,
                                                                initpose, tarposes, camera_intrinsics,
                                                                dathreshold, dawinsize, use_only_da)

    # Return loaded data
    data = {'points': points, 'fwdflows': fwdflows, 'fwdvisibilities': fwdvisibilities,
            'states': states, 'controls': controls, 'poses': poses, 'dt': dt}
    if compute_bwdflows:
        data['masks'] = masks
        data['bwdflows'] = bwdflows
        data['bwdvisibilities'] = bwdvisibilities
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
        self.filter_func = filter_func # Filters samples in the collater

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
        # Filter batch based on custom function
        if self.filter_func is not None:
            filtered_batch = self.filter_func(batch)
        else:
            # Check if there are NaNs in the poses (BWDs compatibility)
            filtered_batch = []
            for sample in batch:
                if sample['poses'].eq(sample['poses']).all():
                    filtered_batch.append(sample)

        # Collate the other samples together using the default collate function
        collated_batch = torch.utils.data.dataloader.default_collate(filtered_batch)

        # Return post-processed batch
        return collated_batch
