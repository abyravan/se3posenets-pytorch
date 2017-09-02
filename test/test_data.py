####### Test data loader for Baxter sequential data
import data
import time
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

options = {'imgHeight': 240, 'imgWidth': 320, 'imgScale': 1e-4, 'imgSuffix': 'sub', 'seqLength': 3, 'stepLength': 2,
           'baxterCtrlType': 'actdiffvel', 'nCtrl': 7}
options['loadDir'] = '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/,,/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_wfixjts_5hrs_Feb10_17/postprocessmotionshalf/'

baxterlabels = data.read_statelabels_file(
    '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/statelabels.txt')
options['meshIds'] = baxterlabels['meshIds']
options['cameraData'] = data.read_cameradata_file(
    '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions/cameradata.txt')

# Get datasets
D = data.read_recurrent_baxter_dataset(options['loadDir'], options['imgSuffix'], options['stepLength'],
                                       options['seqLength'])
fnc = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht=240, img_wd=320, img_scale=1e-4, ctrl_type='actdiffvel',
                                                       mesh_ids=options['meshIds'], camera_data=options['cameraData'])
L = data.BaxterSeqDataset(D, fnc, 'val')

def show_batch(batch):
    # Get a random sample in the batch and display them
    fig, axes = plt.subplots(4, 1)
    fig.show()

    # Show a single example from the batch
    #id = np.random.randint(batch['depths'].size(0))

    # Show all examples of the batch
    for id in xrange(batch['depths'].size(0)):
        # Display depth, labels and flow
        axes[0].set_title("Depth: {}".format(id))
        grid = torchvision.utils.make_grid(batch['depths'][id])
        axes[0].imshow(grid.numpy().transpose(1, 2, 0))

        axes[1].set_title("Labels: {}".format(id))
        grid = torchvision.utils.make_grid(batch['labels'][id])
        npgrid = grid.numpy().transpose(1, 2, 0).astype(np.float32)
        npgrid = (npgrid - npgrid.min()) / (npgrid.max() - npgrid.min())
        axes[1].imshow(npgrid)

        axes[2].set_title("FWD-Flows: {}".format(id))
        grid = torchvision.utils.make_grid(batch['fwdflows'][id])
        npgrid = grid.numpy().transpose(1, 2, 0).astype(np.float32)
        npgrid = (npgrid - npgrid.min()) / (npgrid.max() - npgrid.min())
        axes[2].imshow(npgrid)

        axes[3].set_title("BWD-Flows: {}".format(id))
        grid = torchvision.utils.make_grid(batch['bwdflows'][id])
        npgrid = grid.numpy().transpose(1, 2, 0).astype(np.float32)
        npgrid = (npgrid - npgrid.min()) / (npgrid.max() - npgrid.min())
        axes[3].imshow(npgrid)

        fig.canvas.draw()
        time.sleep(1)
    # if k%8 == 0:
    #	for ax in axes:
    #		ax.cla()

# Create DataLoader
#sampler = SequentialSampler(L)
options['camIntrinsics'] = {'fx': 589.3664541825391/2,
                            'fy': 589.3664541825391/2,
                            'cx': 320.5/2,
                            'cy': 240.5/2}
DataCollater = data.BaxterSeqDatasetCollater(height=240, width=320, intrinsics=options['camIntrinsics'], meshids=options['meshIds'], cuda=False)
trainloader = torch.utils.data.DataLoader(L, batch_size=8, shuffle=True, num_workers=4,
                                          collate_fn=DataCollater.collate_batch) #, sampler = sampler)

for i, sample in enumerate(trainloader):
    print(i, sample['depths'].size(), sample['fwdflows'].size())

    # observe 4th batch and stop.
    if i == 3:
        show_batch(sample)
        break
