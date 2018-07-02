# Torch imports
import torch
import torch.nn as nn
from torch.autograd import Variable

# Global imports
import vtk
import os
import sys
import shutil
import time
import numpy as np

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))


##############
## Function that creates an axes actor, sets a color and x,y,z text
def create_axes_actor(pose, xcol="red", ycol="red", zcol="red",
                      xtext="", ytext="", ztext="", length=0.05):
    # Create a pose
    assert (pose.size() == torch.Size([3, 4]))  # 3x4 matrix
    pose4x4 = torch.eye(4, 4)
    pose4x4[0:3, :] = pose

    # Create a user transform
    transform = vtk.vtkTransform()
    transform.SetMatrix(pose4x4.view(-1).numpy().tolist())

    # Create an axes and set transform
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)

    # Set X,Y,Z axes text
    axes.SetXAxisLabelText(xtext)
    axes.SetYAxisLabelText(ytext)
    axes.SetZAxisLabelText(ztext)

    # Set axes shaft color
    axes.GetXAxisShaftProperty().SetColor(vtk.vtkNamedColors().GetColor3d(xcol))
    axes.GetYAxisShaftProperty().SetColor(vtk.vtkNamedColors().GetColor3d(ycol))
    axes.GetZAxisShaftProperty().SetColor(vtk.vtkNamedColors().GetColor3d(zcol))

    # Set axes arrow color
    axes.GetXAxisTipProperty().SetColor(vtk.vtkNamedColors().GetColor3d(xcol))
    axes.GetYAxisTipProperty().SetColor(vtk.vtkNamedColors().GetColor3d(ycol))
    axes.GetZAxisTipProperty().SetColor(vtk.vtkNamedColors().GetColor3d(zcol))

    # Set length
    axes.SetTotalLength(length,length,length)

    # Return axes
    return axes

######################
#### Setup options
# Parse arguments
import argparse
import configargparse

# parser = configargparse.ArgumentParser(description='Viz transition model results on GT pose data (Baxter)')
# parser.add_argument('--data', type=str, required=True,
#                     help='Path to tar file with pose data')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='Random seed (default: 1)')
# #parser.add_argument('-c', '--checkpoint', required=True,
# #                    help='Path to checkpoint that load data')
# parser.add_argument('--horizon', type=int, default=10, metavar='H',
#                     help='Horizon to predict into the future (default: 10)')
# parser.add_argument('--nexamples', type=int, default=1, metavar='N',
#                     help='Number of examples to visualize (default: 1)')
# largs = parser.parse_args()

largs = argparse.Namespace()
largs.data = 'gtposesctrls-sim.pth.tar'
largs.seed = 100
largs.checkpoint = 'trained_nets/transnets_gtposes/sim-gtposes-default-lr1e-4-rmsprop-cpu/model_best.pth.tar'
largs.horizon = 10

# Set seed
torch.manual_seed(largs.seed)
np.random.seed(largs.seed)

############
## Load data
if largs.data.find('~') != -1:
    largs.data = os.path.expanduser(largs.data)
assert (os.path.exists(largs.data))

# Get the dataset
dataset = torch.load(largs.data)
train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']
print('Dataset size => Train: {}, Val: {}, Test: {}'.format(train_dataset['gtposes'].size(0),
                                                            val_dataset['gtposes'].size(0),
                                                            test_dataset['gtposes'].size(0)))

# Load data and convert to cos/sin representation if needed
gtposes_d  = train_dataset['gtposes']   # B x (seq+1) x state_dim
jtangles_d = train_dataset['jtangles']  # B x (seq+1) x state_dim
controls_d = train_dataset['controls']  # B x seq x 1

### Get the data sequence (jtangles and poses concat, ensure that we don't have different datasets)
# Maybe for now just look at the first few examples...
st, ed = 1000, 101000
gtposes  = gtposes_d[st:ed, 0].clone()
jtangles = jtangles_d[st:ed, 0].clone()

# Save RAM
train_dataset, val_dataset, test_dataset = None, None, None

#########
## Load a saved checkpoint from disk and setup model
if os.path.isfile(largs.checkpoint):
    print("=> loading checkpoint '{}'".format(largs.checkpoint))
    checkpoint = torch.load(largs.checkpoint)
    args = checkpoint['args']
    num_train_iter = checkpoint['epoch'] * args.train_ipe
    print("=> loaded checkpoint (epoch: {}, num train iter: {})".format(checkpoint['epoch'], num_train_iter))
else:
    print("=> no checkpoint found at '{}'".format(largs.checkpoint))
    raise RuntimeError

## Setup network and load pre-trained weights
model = None
if args.model == 'default':
    import ctrlnets
    model = ctrlnets.TransitionModel(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                                     se3_type=args.se3_type, nonlinearity=args.nonlin,
                                     init_se3_iden=args.init_se3_iden, use_kinchain=args.use_kinchain,
                                     delta_pivot='', local_delta_se3=False, use_jt_angles=False)
else:
    assert (False)
model.load_state_dict(checkpoint['model_state_dict']) # Load pre-trained weights

## Set model in eval mode
model.eval()

#########
# Setup a renderer and render window
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName("Axes")
renderWindow.AddRenderer(renderer)

# And an interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Set background
renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("SlateGray"))

# # Render and Start mouse interaction
# renderer.ResetCamera()
# renderWindow.Render()
# renderWindowInteractor.Start()

# # Temp test
# temptest = torch.eye(4)[0:3].clone()
# axes1 = create_axes_actor(temptest)
# renderer.AddActor(axes1)
# renderWindow.Render()

#########
# Get GT poses to render
dt = 1.0/30.0
args.step_len = 2
nsteps = (largs.horizon+1)*args.step_len
id = np.random.randint(0, gtposes.size(0)-nsteps)
poses_gt    = gtposes[id:id+nsteps:args.step_len]
jtangles_gt = jtangles[id:id+nsteps:args.step_len]
ctrls_gt    = (jtangles_gt[1:] - jtangles_gt[:-1]) / (dt * args.step_len)

# Run a forward pass through the pre-trained network
poses_pred = torch.zeros(largs.horizon+1, args.num_se3, 3, 4)
poses_pred[0] = poses_gt[0] # Copy gt poses at t = 0
poses_0 = Variable(poses_gt[0:1].clone()) # 1 x nse3 x 3 x 4
ctrls_t = Variable(ctrls_gt.clone())      # horizon x nctrl
poses_t = []
for k in xrange(largs.horizon):
    pose_i = poses_0 if (k == 0) else poses_t[-1] # Get poses
    ctrl_i = ctrls_t[k:k+1] # Get ctrls
    _, pose_p = model([pose_i, ctrl_i])
    poses_t.append(pose_p)
    poses_pred[k+1] = pose_p.data.clone()

# Run a forward pass, just do 1-step prediction at each step
poses_pred_1step    = torch.zeros(largs.horizon+1, args.num_se3, 3, 4)
poses_pred_1step[0] = poses_gt[0] # Copy gt poses at t = 0
for k in xrange(largs.horizon):
    _, pose_p = model([Variable(poses_gt[k:k+1].clone()),
                       Variable(ctrls_gt[k:k+1].clone())])
    poses_pred_1step[k] = pose_p.data.clone()

# Render the GT and predicted poses
allgtaxes, allpredaxes, allpred1stepaxes = [], [], []
for j in xrange(poses_gt.size(0)):
    gtaxes, predaxes, pred1stepaxes = [], [], []
    for k in xrange(1,poses_gt[k].size(0)): # BG is pose 0, ignore that
        gtaxes.append(create_axes_actor(poses_gt[j,k]))
        predaxes.append(create_axes_actor(poses_pred[j,k], xcol="green", ycol="green", zcol="green"))
        pred1stepaxes.append(create_axes_actor(poses_pred_1step[j,k], xcol="blue", ycol="blue", zcol="blue"))
        renderer.AddActor(gtaxes[-1])
        renderer.AddActor(predaxes[-1])
        renderer.AddActor(pred1stepaxes[-1])
    allgtaxes.append(gtaxes)
    allpredaxes.append(predaxes)
    allpred1stepaxes.append(pred1stepaxes)

# Render and Start mouse interaction
renderer.ResetCamera()
renderWindow.Render()
renderWindowInteractor.Start()