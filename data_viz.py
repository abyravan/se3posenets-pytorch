## Test it
import argparse
args = argparse.Namespace()
#args.data = ['/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/realdata/session_2017-9-2_182257/']
args.data = ['../data/session_2017-9-7_233102/']
args.img_suffix = ''
args.step_len = 2
args.seq_len = 100
args.train_per = 1.0
args.val_per = 0.0
args.ctrl_type = 'actdiffvel'
args.se3_type = 'se3aa'
args.pred_pivot = False
args.num_se3 = 8
args.se2_data = False
args.mean_dt = 0.0697
args.std_dt  = 0.00248

### CSV stuff
import csv

# Read baxter camera data file
def read_intrinsics_file(filename):
    # Read lines in the file
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        label = spamreader.next()
        data = spamreader.next()
        assert(len(label) == len(data))
        for k in xrange(len(label)):
            ret[label[k]] = float(data[k])
    return ret

# Read baxter camera data file
def read_statectrllabels_file(filename):
    # Read lines in the file
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        statenames = spamreader.next()[:-1]
        ctrlnames  = spamreader.next()[:-1]
        try:
            trackernames = spamreader.next()[:-1]
        except:
            trackernames = []
    return statenames, ctrlnames, trackernames

# Read baxter joint labels and their corresponding mesh index value
def read_statelabels_file(filename):
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader      = csv.reader(csvfile, delimiter=' ')
        ret['frames']   = spamreader.next()[0:-1]
        ret['meshIds']  = [int(x) for x in spamreader.next()[0:-1]]
    return ret

# Read baxter state files
def read_baxter_state_file(filename):
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        ret['actjtpos']     = spamreader.next()[0:-1] # Last element is a string due to the way the file is created
        ret['actjtvel']     = spamreader.next()[0:-1]
        ret['actjteff']     = spamreader.next()[0:-1]
        ret['comjtpos']     = spamreader.next()[0:-1]
        ret['comjtvel']     = spamreader.next()[0:-1]
        ret['comjtacc']     = spamreader.next()[0:-1]
        ret['tarendeffpos'] = spamreader.next()[0:-1]
    return ret

###
# Get default options & camera intrinsics
load_dir = args.data[0] #args.data.split(',,')[0]
try:
    # Read from file
    intrinsics = read_intrinsics_file(load_dir + "/intrinsics.txt")
    print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
    if args.se2_data:
        args.img_ht, args.img_wd = int(intrinsics['ht']), int(intrinsics['wd'])
    else:
        args.img_ht, args.img_wd = 240, 320  # All data except SE(2) data is at 240x320 resolution
    args.img_scale = 1.0 / intrinsics['s']  # Scale of the image (use directly from the data)

    # Setup camera intrinsics
    sc = float(args.img_ht) / intrinsics['ht']  # Scale factor for the intrinsics
    args.cam_intrinsics = {'fx': intrinsics['fx'] * sc,
                           'fy': intrinsics['fy'] * sc,
                           'cx': intrinsics['cx'] * sc,
                           'cy': intrinsics['cy'] * sc}
    print("Scale factor for the intrinsics: {}".format(sc))
except:
    print("Could not read intrinsics file, reverting to default settings")
    args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
    args.cam_intrinsics = {'fx': 589.3664541825391 / 2,
                           'fy': 589.3664541825391 / 2,
                           'cx': 320.5 / 2,
                           'cy': 240.5 / 2}
print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(args.img_ht, args.img_wd,
                                                                            args.cam_intrinsics['fx'],
                                                                            args.cam_intrinsics['fy'],
                                                                            args.cam_intrinsics['cx'],
                                                                            args.cam_intrinsics['cy']))

# Get dimensions of ctrl & state
try:
    statelabels, ctrllabels, trackerlabels = read_statectrllabels_file(load_dir + "/statectrllabels.txt")
    print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
except:
    statelabels = read_statelabels_file(load_dir + '/statelabels.txt')['frames']
    ctrllabels = statelabels  # Just use the labels
    print("Could not read statectrllabels file. Reverting to labels in statelabels file")
args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
print('Num state: {}, Num ctrl: {}, Num tracker: {}'.format(args.num_state, args.num_ctrl, args.num_tracker))

## TODO: More general
statefile = load_dir + "/motion0/state0.txt"
state = read_baxter_state_file(statefile)

# Get number of images
postprocessstatsfile = load_dir + "/motion0/postprocessstats.txt"
with open(postprocessstatsfile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
    nimages = int(spamreader.next()[0])
print("Number of images: {}".format(nimages))

#########
# Load pangolin visualizer library (before torch)
import _init_paths
#import numpy as np
from torchviz import pangodataviz
pangolin = pangodataviz.PyPangolinDataViz(args.data[0], nimages, args.step_len, args.seq_len, args.num_se3,
                                         args.img_ht, args.img_wd, args.num_state, args.num_ctrl,
                                         args.num_tracker,
                                         args.cam_intrinsics['fx'],
                                         args.cam_intrinsics['fy'],
                                         args.cam_intrinsics['cx'],
                                         args.cam_intrinsics['cy'])
                                         #np.asarray(state['actjtpos'], dtype=np.float32))

#########
# Torch imports
import torch
import cv2
import data as datav
import ctrlnets

# Find the IDs of the controlled joints in the state vector
# We need this if we have state dimension > ctrl dimension and
# if we need to choose the vals in the state vector for the control
ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
ctrlids_in_track = torch.LongTensor([trackerlabels.index(x) for x in ctrllabels])
print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))
print("ID of controlled joints in the track vector: ", ctrlids_in_track.view(1, -1))

# Compute intrinsic grid
args.cam_intrinsics['xygrid'] = datav.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                            args.cam_intrinsics)

# Image suffix
args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

# Read mesh ids and camera data
args.baxter_labels = datav.read_statelabels_file(load_dir + '/statelabels.txt')
args.mesh_ids      = args.baxter_labels['meshIds']
args.cam_extrinsics = datav.read_cameradata_file(load_dir + '/cameradata.txt')
print(args.cam_extrinsics['modelView'])

# SE3 stuff
assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
args.se3_dim = ctrlnets.get_se3_dimension(args.se3_type, args.pred_pivot)
print('Predicting {} SE3s of type: {}. Dim: {}'.format(args.num_se3, args.se3_type, args.se3_dim))

# Sequence stuff
print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

########################
############ Load datasets
# Get datasets
baxter_data     = datav.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                     step_len = args.step_len, seq_len = args.seq_len,
                                                     train_per = args.train_per, val_per = args.val_per)
disk_read_func  = lambda d, i: datav.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                   img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                                   num_ctrl=args.num_ctrl, num_state=args.num_state,
                                                                   mesh_ids = args.mesh_ids, ctrl_ids=ctrlids_in_state,
                                                                   camera_extrinsics = args.cam_extrinsics,
                                                                   camera_intrinsics = args.cam_intrinsics,
                                                                   compute_bwdflows=True, load_color=True,
                                                                   num_tracker=args.num_tracker)
filter_func = lambda b: datav.filter_func(b, mean_dt=args.mean_dt, std_dt=args.std_dt)
train_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'train', filter_func) # Train dataset
print('Dataset size => Train: {}'.format(len(train_dataset)))

########################
############ Load based on pangolin input
id = torch.zeros(1)
while True:
    # Read data
    print("Reading data....")
    sample = train_dataset[int(id[0])]

    # Compute act, com & dart diff vels
    actdiffvels = sample['controls']
    comdiffvels = (sample['comconfigs'][1:] - sample['comconfigs'][:-1]) / sample['dt']
    dartdiffvels = (sample['trackerconfigs'][1:][:, ctrlids_in_track] -
                    sample['trackerconfigs'][:-1][:, ctrlids_in_track]) / sample['dt'] # state -> ctrl dimension


    # Send data to pangolin & get ID
    print("Sending data to pangolin....")
    pangolin.update_viz(sample['points'].numpy(),
                        sample['fwdflows'].numpy(),
                        sample['bwdflows'].numpy(),
                        sample['fwdvisibilities'].byte().numpy(),
                        sample['bwdvisibilities'].byte().numpy(),
                        sample['labels'].byte().numpy(),
                        sample['rgbs'].permute(0,2,3,1).clone().numpy(),
                        sample['masks'].byte().numpy(),
                        sample['poses'].numpy(),
                        sample['poses'].numpy(),                    #TODO: Fix
                        args.cam_extrinsics['modelView'].float().numpy(),
                        sample['actconfigs'].numpy(),
                        sample['actvels'].numpy(),
                        sample['comconfigs'].numpy(),
                        sample['comvels'].numpy(),
                        sample['trackerconfigs'].numpy(),
                        actdiffvels.numpy(),
                        comdiffvels.numpy(),
                        dartdiffvels.numpy(),
                        sample['controls'].numpy(),
                        id.numpy())

'''
# Display some stuff
sample = train_dataset[20]

import matplotlib.pyplot as plt
import torchvision
depthdisp = torchvision.utils.make_grid(sample['points'].narrow(1,2,1).clone(), normalize=True, range=(0.0,3.0))
fig = plt.figure(100)
plt.imshow(depthdisp.permute(1,2,0).numpy())

flowdisp = torchvision.utils.make_grid(sample['fwdflows'].clone(), normalize=True, range=(-0.05,0.05))
fig1 = plt.figure(101)
plt.imshow(flowdisp.permute(1,2,0).numpy())

maskdisp = torchvision.utils.make_grid(sample['masks'][0:1].view(-1,1,args.img_ht,args.img_wd).float().clone(), normalize=True, range=(0.0,1.0))
fig2 = plt.figure(102)
plt.imshow(maskdisp.permute(1,2,0).numpy())
'''