#!/usr/bin/env python

############# Import pangolin
# TODO: Make this cleaner, we don't need most of these parameters to create the pangolin window
img_ht, img_wd, img_scale = 240, 320, 1e-4
seq_len = 1 # For now, only single step
num_se3 = 8 #20 # TODO: Especially this parameter!
dt = 1.0/30.0
oldgrippermodel = False # TODO: When are we actually going to use the new ones?
cam_intrinsics = {'fx': 589.3664541825391 / 2,
                  'fy': 589.3664541825391 / 2,
                  'cx': 320.5 / 2,
                  'cy': 240.5 / 2}
savedir = 'temp' # TODO: Fix this!

# Load pangolin visualizer library
import _init_paths
from torchviz import realctrlcompviz
pangolin = realctrlcompviz.PyRealCtrlCompViz(img_ht, img_wd, img_scale, num_se3,
                                         cam_intrinsics['fx'], cam_intrinsics['fy'],
                                         cam_intrinsics['cx'], cam_intrinsics['cy'],
                                         savedir, 0)


# Global imports
import argparse
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

##########
# NOTE: When importing torch before initializing the pangolin window, I get the error:
#   Pangolin X11: Unable to retrieve framebuffer options
# Long story short, initializing torch before pangolin messes things up big time.
# Also, only the conda version of torch works otherwise there's an issue with loading the torchviz library before torch
#   ImportError: dlopen: cannot load any more object with static TLS
# With the new CUDA & NVIDIA drivers the new conda also doesn't work. had to move to CYTHON to get code to work

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision
import cv2
import threading

# Local imports
import se3layers as se3nn
import data
import ctrlnets
import util
from util import AverageMeter, Tee

# Read data
gndataall = torch.load('comp-gn-sim/datastats.pth.tar')
bpdataall = torch.load('comp-backprop-sim/datastats.pth.tar')

###################### RE-RUN VIZ TO SAVE FRAMES TO DISK CORRECTLY
######## Saving frames to disk now!
for k in [0,1,5]: #xrange(len(gndataall)):
    gndata = gndataall[k]
    bpdata = bpdataall[k]
    initstats = gndata[0]

    curr_angles_s = {'gn': gndata[1], 'backprop': bpdata[1]}
    curr_pts_s    = {'gn': gndata[2], 'backprop': bpdata[2]}
    curr_poses_s  = {'gn': gndata[3], 'backprop': bpdata[3]}
    curr_masks_s  = {'gn': gndata[4], 'backprop': bpdata[4]}
    curr_rgb_s    = {'gn': gndata[5], 'backprop': bpdata[5]}
    loss_s        = {'gn': gndata[6], 'backprop': bpdata[6]}
    err_indiv_s   = {'gn': gndata[7], 'backprop': bpdata[7]}
    curr_deg_errors_s = {'gn': gndata[8], 'backprop': bpdata[8]}

    pangolin.update_real_init(initstats[0],
                              initstats[1],
                              initstats[2],
                              initstats[3],
                              initstats[4],
                              initstats[5],
                              initstats[6],
                              initstats[7],
                              initstats[8],
                              initstats[9],
                              initstats[10],
                              initstats[11],
                              initstats[12])

    # Start saving frames
    save_dir = "comp-both/frames/test" + str(int(k)) + "/"
    util.create_dir(save_dir)  # Create directory
    pangolin.start_saving_frames(save_dir)  # Start saving frames

    mx = max(len(curr_angles_s['gn']), len(curr_angles_s['backprop']))
    for j in xrange(mx):
        jg = j if (j < len(curr_angles_s['gn'])) else len(curr_angles_s['gn'])-1
        jb = j if (j < len(curr_angles_s['backprop'])) else len(curr_angles_s['backprop'])-1
        if (j%10  == 0):
            print("Saving frame: {}/{}".format(j, mx))
        pangolin.update_real_curr(curr_angles_s['gn'][jg].numpy(),
                                  curr_pts_s['gn'][jg].numpy(),
                                  curr_poses_s['gn'][jg].numpy(),
                                  curr_masks_s['gn'][jg].numpy(),
                                  curr_rgb_s['gn'][jg].numpy(),
                                  loss_s['gn'][jg],
                                  err_indiv_s['gn'][jg].numpy(),
                                  curr_deg_errors_s['gn'][jg].numpy(),
                                  curr_angles_s['backprop'][jb].numpy(),
                                  curr_pts_s['backprop'][jb].numpy(),
                                  curr_poses_s['backprop'][jb].numpy(),
                                  curr_masks_s['backprop'][jb].numpy(),
                                  curr_rgb_s['backprop'][jb].numpy(),
                                  loss_s['backprop'][jb],
                                  err_indiv_s['backprop'][jb].numpy(),
                                  curr_deg_errors_s['backprop'][jb].numpy(),
                                  0) # Save frame
        time.sleep(0.1)

    # Stop saving frames
    time.sleep(5)
    pangolin.stop_saving_frames()