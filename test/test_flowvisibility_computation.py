import torch
import cv2
import data as datav
from layers._ext import se3layers

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
    sequence = datav.generate_baxter_sequence(dataset, id)  # Get the file paths
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
        depths[k] = datav.read_depth_image(s['depth'], img_ht, img_wd, img_scale) # Third channel is depth (x,y,z)

        # Load label
        labels[k] = torch.ByteTensor(cv2.imread(s['label'], -1)) # Put the masks in the first channel

        # Load configs
        state = datav.read_baxter_state_file(s['state1'])
        actconfigs[k] = state['actjtpos']
        comconfigs[k] = state['comjtpos']

        # Load SE3 state & get all poses
        se3state = datav.read_baxter_se3state_file(s['se3state1'])
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

    # Try to compute the flows and visibility
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
    poseinvs_1 = datav.RtInverse(poses_1.clone())
    poseinvs_2 = datav.RtInverse(poses_2.clone())

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

'''
### Compute fwd/bwd visibility (for points in time t, which are visible in time t+1 & vice-versa)
def ComputeFlowAndVisibility(cloud_1, cloud_2, label_1, label_2,
                             poses_1, poses_2, intrinsics,
                             dathreshold=0.01, dawinsize=5):
    # Create memory
    bsz, seq, dim, ht, wd = cloud_1.size()
    fwdflows      = torch.FloatTensor(bsz, seq, 3, ht, wd).type_as(cloud_1)
    bwdflows      = torch.FloatTensor(bsz, seq, 3, ht, wd).type_as(cloud_2)
    fwdvisibility = torch.ByteTensor(bsz, seq, 1, ht, wd).cuda() if cloud_1.is_cuda else torch.ByteTensor(bsz, seq, 1, ht, wd)
    bwdvisibility = torch.ByteTensor(bsz, seq, 1, ht, wd).cuda() if cloud_2.is_cuda else torch.ByteTensor(bsz, seq, 1, ht, wd)

    # Compute inverse of poses
    poseinvs_1 = datav.RtInverse(poses_1.view(bsz*seq, -1, 3, 4))
    poseinvs_2 = datav.RtInverse(poses_2.view(bsz*seq, -1, 3, 4))

    # Call cpp/CUDA functions
    if cloud_1.is_cuda:
        se3layers.ComputeFlowAndVisibility_cuda(cloud_1.view(bsz*seq, 3, ht, wd),
                                                cloud_2.view(bsz*seq, 3, ht, wd),
                                                label_1.view(bsz*seq, 1, ht, wd),
                                                label_2.view(bsz*seq, 1, ht, wd),
                                                poses_1.view(bsz*seq, -1, 3, 4),
                                                poses_2.view(bsz*seq, -1, 3, 4),
                                                poseinvs_1.view(bsz*seq, -1, 3, 4),
                                                poseinvs_2.view(bsz*seq, -1, 3, 4),
                                                fwdflows.view(bsz*seq, 3, ht, wd),
                                                bwdflows.view(bsz*seq, 3, ht, wd),
                                                fwdvisibility.view(bsz*seq, 1, ht, wd),
                                                bwdvisibility.view(bsz*seq, 1, ht, wd),
                                                intrinsics['fx'],
                                                intrinsics['fy'],
                                                intrinsics['cx'],
                                                intrinsics['cy'],
                                                dathreshold,
                                                dawinsize)
    else:
        assert(cloud_1.type() == 'torch.FloatTensor')
        se3layers.ComputeFlowAndVisibility_float(cloud_1.view(bsz*seq, 3, ht, wd),
                                                cloud_2.view(bsz*seq, 3, ht, wd),
                                                label_1.view(bsz*seq, 1, ht, wd),
                                                label_2.view(bsz*seq, 1, ht, wd),
                                                poses_1.view(bsz*seq, -1, 3, 4),
                                                poses_2.view(bsz*seq, -1, 3, 4),
                                                poseinvs_1.view(bsz*seq, -1, 3, 4),
                                                poseinvs_2.view(bsz*seq, -1, 3, 4),
                                                fwdflows.view(bsz*seq, 3, ht, wd),
                                                bwdflows.view(bsz*seq, 3, ht, wd),
                                                fwdvisibility.view(bsz*seq, 1, ht, wd),
                                                bwdvisibility.view(bsz*seq, 1, ht, wd),
                                                intrinsics['fx'],
                                                intrinsics['fy'],
                                                intrinsics['cx'],
                                                intrinsics['cy'],
                                                dathreshold,
                                                dawinsize)

    # Return
    return fwdflows, bwdflows, fwdvisibility, bwdvisibility
'''

## Test it
import argparse
args = argparse.Namespace()
args.data = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions_f/,,'\
            '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_wfixjts_5hrs_Feb10_17/postprocessmotionshalf_f/,,'\
            '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint0/postprocessmotions_f/,,'\
            '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint5/postprocessmotions_f/'
args.img_suffix = 'sub'
args.step_len = 2
args.seq_len = 10
args.train_per = 0.6
args.val_per = 0.15
args.img_ht = 240
args.img_wd = 320
args.img_scale = 1e-4
args.ctrl_type = 'actdiffvel'
args.batch_size = 16
args.use_pin_memory = False
args.num_workers = 6
args.cuda = True

# Read mesh ids and camera data
load_dir = args.data.split(',,')[0]
args.baxter_labels = datav.read_statelabels_file(load_dir + '/statelabels.txt')
args.mesh_ids      = args.baxter_labels['meshIds']
args.cam_extrinsics = datav.read_cameradata_file(load_dir + '/cameradata.txt')
args.cam_intrinsics = {'fx': 589.3664541825391/2,
                       'fy': 589.3664541825391/2,
                       'cx': 320.5/2,
                       'cy': 240.5/2}
args.cam_intrinsics['xygrid'] = datav.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                           args.cam_intrinsics)

# Test
baxter_data     = datav.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                     step_len = args.step_len, seq_len = args.seq_len,
                                                     train_per = args.train_per, val_per = args.val_per)



#############
# Test function 1
disk_read_func  = lambda d, i: datav.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                    img_scale = args.img_scale, ctrl_type = 'actdiffvel',
                                                                    mesh_ids = args.mesh_ids,
                                                                    camera_extrinsics = args.cam_extrinsics,
                                                                    camera_intrinsics = args.cam_intrinsics,
                                                                    compute_bwdflows=True) # No need for BWD flows
train_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset
print('Dataset size => Train: {}'.format(len(train_dataset)))

# Sample indices
num = 1
ids = torch.IntTensor([10000])

# Test speed
import time
st = time.time()
for k in xrange(num):
    sample = train_dataset[ids[k]]
tot = time.time() - st
print('[OLD] Total time: {}, Avg time: {}'.format(tot, tot/num))

#############
# Test function 2
disk_read_func_1  = lambda d, i: read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                              img_scale = args.img_scale, ctrl_type = 'actdiffvel',
                                                              mesh_ids = args.mesh_ids,
                                                              camera_extrinsics = args.cam_extrinsics,
                                                              camera_intrinsics = args.cam_intrinsics,
                                                              compute_bwdflows=True) # No need for BWD flows
train_dataset_1 = datav.BaxterSeqDataset(baxter_data, disk_read_func_1, 'train') # Train dataset
print('Dataset size => Train: {}'.format(len(train_dataset_1)))

# Test speed
#ids = (torch.rand(num) * (len(train_dataset)-1)).round().int()
st1 = time.time()
for k in xrange(num):
    sample1 = train_dataset_1[ids[k]]
tot1 = time.time() - st1
print('[NEW] Total time: {}, Avg time: {}'.format(tot1, tot1/num))


'''
import time
st = time.time()
num = 100
for k in xrange(num):
    # Try to compute the flows and visibility
    tarpts = sample['points'].unsqueeze(0)[:, 1:]  # t+1, t+2, t+3, ....
    initpt = sample['points'].unsqueeze(0)[:, 0:1].expand_as(tarpts).clone()
    tarlabels = sample['labels'].unsqueeze(0)[:, 1:]  # t+1, t+2, t+3, ....
    initlabel = sample['labels'].unsqueeze(0)[:, 0:1].expand_as(tarlabels).clone()
    tarposes = sample['allposes'].unsqueeze(0)[:, 1:]  # t+1, t+2, t+3, ....
    initpose = sample['allposes'].unsqueeze(0)[:, 0:1].expand_as(tarposes).clone()

    # Compute flow and visibility
    f, b, fv, bv = ComputeFlowAndVisibility(initpt, tarpts,
                                            initlabel, tarlabels,
                                            initpose, tarposes,
                                            args.cam_intrinsics,
                                            args.mesh_ids)

tot = time.time() - st
print('Total time: {}, Avg time: {}'.format(tot, tot/num))

diff = fv.float() - sample['fwdvisibilities'].float()
print(diff.max(), diff.min(), diff.eq(0).sum(), diff.ne(0).sum())
print(fv.eq(0).sum(), sample['fwdvisibilities'].eq(0).sum())

import matplotlib.pyplot as plt
fig = plt.figure(100)
fig.show()
plt.imshow(fv[0,2,0].numpy())
fig1 = plt.figure(101)
plt.imshow(sample['fwdvisibilities'][2,0].numpy())
fig2 = plt.figure(102)
for k in xrange(5):
    fig2 = plt.figure(100+k)
    plt.imshow(diff[0,k,0].numpy())

for k in xrange(5):
    fig2 = plt.figure(100+k)
    plt.imshow(fv[0,k,0].numpy())

difff = f.float() - sample['fwdflows'].float()
diffb = b.float() - sample['bwdflows'].float()
print('Fwd: ', difff.max(), difff.min())
print('Bwd: ', diffb.max(), diffb.min())
'''