#########
# Torch imports
import torch
import cv2
import data as datav
import ctrlnets
import util

## Test it
import argparse
args = argparse.Namespace()
root_dir = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/realdata/'
args.data = [root_dir + '/session_2017-9-7_233102/',
             root_dir + '/session_2017-9-8_02519/',
             root_dir + '/session_2017-9-8_111150/',
             root_dir + '/session_2017-9-8_134717/',
             root_dir + 'session_2017-9-8_152749/']
args.img_suffix = ''
args.step_len = 2
args.seq_len = 5
args.train_per = 0.7
args.val_per = 0.1
args.ctrl_type = 'actdiffvel'
args.se3_type = 'se3aa'
args.pred_pivot = False
args.num_se3 = 8
args.se2_data = False
args.batch_size = 16
args.num_workers = 8
args.use_pin_memory = False
args.da_winsize = 5
args.da_threshold = 0.015
args.use_only_da_for_flows = False
args.seed = 1

args.cuda=True
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

###
# Get default options & camera intrinsics
load_dir = args.data[0] #args.data.split(',,')[0]
try:
    # Read from file
    intrinsics = datav.read_intrinsics_file(load_dir + "/intrinsics.txt")
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
    statelabels, ctrllabels, trackerlabels = datav.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
    print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
except:
    statelabels = datav.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
    ctrllabels = statelabels  # Just use the labels
    trackerlabels = []
    print("Could not read statectrllabels file. Reverting to labels in statelabels file")
args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))


# Find the IDs of the controlled joints in the state vector
# We need this if we have state dimension > ctrl dimension and
# if we need to choose the vals in the state vector for the control
ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

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
                                                                   compute_bwdflows=False, num_tracker=args.num_tracker,
                                                                   dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                                   use_only_da=args.use_only_da_for_flows)
train_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset

'''
print('Dataset size => Train: {}'.format(len(train_dataset)))
train_loader = util.DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.use_pin_memory,
                                                    collate_fn=train_dataset.collate_batch))
sample = {}
while train_loader.iteration_count() <= 91501:
    i, sample = train_loader.next()
    if (train_loader.iteration_count() % 1000 == 0):
        print("{}/{}".format(train_loader.iteration_count(), 91501))
print(sample['id'])
torch.save(sample['id'], "temp.pth.tar")
'''

# Read data
samples = []
ids = torch.load('temp.pth.tar')
for k in ids:
    samples.append(train_dataset[k])

# Display stuff
import matplotlib.pyplot as plt
import torchvision
import time
plt.ion()
for k in xrange(len(samples)):
    sample = samples[k]
    depths = sample['points'].narrow(1,2,1).clone()
    print(k, depths.max(), depths.min())
    depthdisp = torchvision.utils.make_grid(depths, nrow=args.seq_len+1, normalize=True, range=(0.0,3.0))
    fig = plt.figure(100)
    plt.title("ID: {}".format(k))
    plt.imshow(depthdisp.permute(1,2,0).numpy())
    fig.show()
    plt.pause(0.01)
    time.sleep(2)
