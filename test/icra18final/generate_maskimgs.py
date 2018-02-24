# Global imports
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Torch imports
import torch.utils.data
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append("/home/barun/Projects/se3nets-pytorch/")

# Local imports
import ctrlnets
import se3nets
import util
import cv2

#### Setup options
# Common
import configargparse

# Loss options
parser = configargparse.ArgumentParser(description='SE3-Pose-Nets Mask rendering')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', required=True,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='', type=str, metavar='PATH', required=True,
                    help='path to saved torch data containing input 3D points (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH', required=True,
                    help='save directory (default: none)')
parser.add_argument('--net-type', default='', type=str, metavar='PATH', required=True,
                    help='se3 | se3pose (default: none)')
pargs = parser.parse_args()

##################
##### Load saved disk network
print("Loading pre-trained network from: {}".format(pargs.checkpoint))
checkpoint = torch.load(pargs.checkpoint)
args       = checkpoint['args']
num_train_iter   = checkpoint['train_iter']

##################
##### Load network
print("Initializing network and copying parameters")
num_input_channels = 3  # Num input channels
if args.use_xyzrgb:
    num_input_channels = 6
if pargs.net_type == 'se3pose' or pargs.net_type == 'se3posekinchain':
    model = ctrlnets.MultiStepSE3PoseModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                    se3_type=args.se3_type, delta_pivot=args.delta_pivot, use_kinchain=args.local_delta_se3,
                    input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                    init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                    use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                    sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv, decomp_model=args.decomp_model,
                    use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                    wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                    use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state_net,
                    full_res=args.full_res,
                    noise_stop_iter=args.noise_stop_iter)  # noise_stop_iter not available for SE2 models
elif pargs.net_type == 'se3':
    model = se3nets.SE3Model(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                             se3_type=args.se3_type, use_pivot=args.pred_pivot, use_kinchain=False,
                             input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                             init_transse3_iden=args.init_transse3_iden,
                             use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                             sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                             wide=args.wide_model, se2_data=False, use_jt_angles=args.use_jt_angles,
                             num_state=args.num_state_net, use_lstm=(args.seq_len > 1))
else:
    assert False, "Unknown model type input: {}".format(pargs.net_type)
if args.cuda:
    model.cuda()  # Convert to CUDA if enabled

# Load parameters from saved checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Test mode

##################
##### Load data
print("Loading data from: {}".format(pargs.data))
loaddata = torch.load(pargs.data)
samples = loaddata['samples']

# Iterate over samples and make predictions
deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'  # Default tensor type
netinputs_a, ctrls_a, jtangles_a, predmasks_a, gtmasks_a = [], [], [], [], []
util.create_dir(pargs.save_dir)
for k in range(len(samples)):
    # Stats
    if (k+1)%10 == 0:
        print("Iterating over sample: {}/{}".format(k, len(samples)))

    # Get sample
    sample = samples[k]

    ## Load data
    pts   = util.to_var(sample['points'].type(deftype), volatile=True)
    ctrls = util.to_var(sample['controls'].type(deftype))
    gtmask = util.to_var(sample['masks'].type(deftype))

    # Get XYZRGB input
    if args.use_xyzrgb:
        rgb = util.to_var(sample['rgbs'].type(deftype) / 255.0)  # Normalize RGB to 0-1
        netinput = torch.cat([pts, rgb], 1)  # Concat along channels dimension
    elif args.use_xyzhue:
        hue = util.to_var(sample['rgbs'].narrow(2, 0, 1).type(deftype) / 179.0)  # Normalize Hue to 0-1 (Opencv has hue from 0-179)
        netinput = torch.cat([pts, hue], 1)  # Concat along channels dimension
    else:
        netinput = pts  # XYZ

    # Get jt angles
    if args.box_data:
        jtangles = util.to_var(sample['states'].type(deftype))
    else:
        jtangles = util.to_var(sample['actctrlconfigs'].type(deftype))

    ##################
    ##### Run forward pass to predict masks
    if pargs.net_type == 'se3pose' or pargs.net_type == 'se3posekinchain':
        _, initmask = model.forward_pose_mask([netinput[0:1], jtangles[0:1]], train_iter=num_train_iter)
    else:
        _, [_, initmask] = model([netinput[0:1], jtangles[0:1], ctrls[0:1]],
                              reset_hidden_state=True,
                              train_iter=num_train_iter)  # Reset hidden state at start of sequence

    ##################
    ##### Save all data
    netinputs_a.append(netinput[0:1].data.cpu().clone())
    ctrls_a.append(ctrls[0:1].data.cpu().clone())
    jtangles_a.append(jtangles[0:1].data.cpu().clone())
    predmasks_a.append(initmask.data.cpu().clone())
    gtmasks_a.append(gtmask[0:1].data.cpu().clone())

    ##### GT labels
    _, labels = gtmask.data.max(dim=1)
    cv2.imwrite(labels.cpu().squeeze().clone().numpy(), pargs.save_dir + "/mask{}.png".format(k))

##### Save masks
print("Saving results at: {}".format(pargs.save_dir + "maskresults.pth.tar"))
savedata = {'netinputs': netinputs_a, 'ctrls': ctrls_a, 'jtangles': jtangles_a, 'predmasks': predmasks_a,
            'gtmasks': gtmasks_a, 'pargs': pargs, 'args': 'args', 'sampleids': loaddata['ids']}
torch.save(loaddata, pargs.save_dir + "/maskresults.pth.tar")