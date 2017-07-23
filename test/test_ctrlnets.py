import ctrlnets
import torch
import time
from torch.autograd import Variable
import torch.nn as nn
import se3layers as se3nn

#### Test SE3-Pose-Model
bsz, ht, wd, nse3, nctrl = 4, 240, 320, 8, 7
se3type = 'se3aa'
model = ctrlnets.SE3PoseModel(num_ctrl=nctrl, num_se3=nse3, se3_type=se3type,
                              use_pivot=False, input_channels=3, use_bn=True,
                              use_kinchain=False, nonlinearity='prelu')
for k in xrange(10):
    # Generate data
    ptcloud_0, ptcloud_1, ctrls = torch.rand(bsz, 3, ht, wd), torch.rand(bsz, 3, ht, wd), torch.rand(bsz, nctrl)
    tgtptcloud_0, tgtptcloud_1 = torch.rand(bsz, 3, ht, wd), torch.rand(bsz, 3, ht, wd)
    ptcloud_0_var = Variable(ptcloud_0, requires_grad=True)
    ptcloud_1_var = Variable(ptcloud_1, requires_grad=True)
    ctrls_var = Variable(ctrls, requires_grad=True)
    tgtptcloud_0_var = Variable(tgtptcloud_0, requires_grad=False)
    tgtptcloud_1_var = Variable(tgtptcloud_1, requires_grad=False)

    # Run forward pass
    [pose_0, mask_0], [pose_1, mask_1], [deltapose_t_01, pose_t_1] = model([ptcloud_0_var,
                                                                            ptcloud_1_var,
                                                                            ctrls_var])
    deltapose_t_10 = se3nn.RtInverse()(deltapose_t_01) # Invert the delta-pose

    # Now compute 3D loss @ t1 & t2
    ptloss_0 = se3nn.Weighted3DTransformLoss()(ptcloud_0_var, mask_0, deltapose_t_01, tgtptcloud_0_var) # Predict pts in FWD dirn and compare to target @ t1
    ptloss_1 = se3nn.Weighted3DTransformLoss()(ptcloud_1_var, mask_1, deltapose_t_10, tgtptcloud_1_var) # Predict pts in BWD dirn and compare to target @ t0
    poseloss = ctrlnets.BiMSELoss(pose_1, pose_t_1) # Enforce consistency between pose @ t1 predicted by encoder & pose @ t1 from transition model

    # Total loss is sum of these
    loss = ptloss_0 + ptloss_1 + 0.01 * poseloss
    model.zero_grad()  # Zero model gradients
    loss.backward() # Compute BWD

    # Print some grad data
    print "Iter: {}".format(k)
    print ptcloud_0_var.grad.max(), ptcloud_0_var.grad.min()
    print ptcloud_1_var.grad.max(), ptcloud_1_var.grad.min()
    print ctrls_var.grad.max(), ctrls_var.grad.min()

time.sleep(100)

#### Test entire setup
bsz, ht, wd, nse3, nctrl = 4, 240, 320, 8, 7
se3type = 'se3aa'

# Create the pose-mask encoder
M = ctrlnets.PoseMaskEncoder(num_se3=nse3, se3_type=se3type, use_pivot=False,
                                 use_kinchain=False, input_channels=3, use_bn=True, nonlinearity='prelu')
M.cuda()

# Create the transition model
T = ctrlnets.TransitionModel(num_ctrl=nctrl, num_se3=nse3, se3_type=se3type, use_pivot=False,
                                 use_kinchain=False, nonlinearity='prelu')
T.cuda()

# Run a loop to see how it works
for k in xrange(10):
    # Generate data
    ptcloud_0, ptcloud_1, ctrls = torch.rand(bsz, 3, ht, wd), torch.rand(bsz, 3, ht, wd), torch.rand(bsz, nctrl)
    tgtptcloud_0, tgtptcloud_1  = torch.rand(bsz, 3, ht, wd), torch.rand(bsz, 3, ht, wd)
    ptcloud_0_var = Variable(ptcloud_0.cuda(), requires_grad=True)
    ptcloud_1_var = Variable(ptcloud_1.cuda(), requires_grad=True)
    ctrls_var     = Variable(ctrls.cuda(), requires_grad=True)
    tgtptcloud_0_var = Variable(tgtptcloud_0.cuda(), requires_grad=False)
    tgtptcloud_1_var = Variable(tgtptcloud_1.cuda(), requires_grad=False)

    # Get pose & mask predictions @ t0 & t1
    pose_0, mask_0 = M(ptcloud_0_var) # ptcloud @ t0
    pose_1, mask_1 = M(ptcloud_1_var) # ptcloud @ t1

    # Get transition model predicton of pose_1
    deltapose_t_01, pose_t_1 = T([pose_0, ctrls_var]) # Predicts [delta-pose, pose]
    deltapose_t_10 = se3nn.RtInverse()(deltapose_t_01) # Invert the delta-pose

    # Now compute 3D loss @ t1 & t2
    ptloss_0 = se3nn.Weighted3DTransformLoss()(ptcloud_0_var, mask_0, deltapose_t_01, tgtptcloud_0_var) # Predict pts in FWD dirn and compare to target @ t1
    ptloss_1 = se3nn.Weighted3DTransformLoss()(ptcloud_1_var, mask_1, deltapose_t_10, tgtptcloud_1_var) # Predict pts in BWD dirn and compare to target @ t0
    poseloss = ctrlnets.BiMSELoss(pose_1, pose_t_1) # Enforce consistency between pose @ t1 predicted by encoder & pose @ t1 from transition model

    # Total loss is sum of these
    loss = ptloss_0 + ptloss_1 + 0.01 * poseloss
    M.zero_grad(); T.zero_grad() # Zero model gradients
    loss.backward() # Compute BWD

    # Print some grad data
    print "Iter: {}".format(k)
    print ptcloud_0_var.grad.max(), ptcloud_0_var.grad.min()
    print ptcloud_1_var.grad.max(), ptcloud_1_var.grad.min()
    print ctrls_var.grad.max(), ctrls_var.grad.min()

time.sleep(100)

#### Test transition model
print('==> Testing transition model <==')
T = ctrlnets.TransitionModel(num_ctrl=7, num_se3=8, se3_type='se3aa', use_pivot=False,
                                 use_kinchain=False,nonlinearity='prelu')
T.cuda()
for k in xrange(10):
    pose,ctrl = Variable(torch.rand(8,8,3,4).cuda()), Variable(torch.rand(8,7).cuda())
    st = time.time()
    output = T([pose,ctrl])
    print('Time for FWD pass: {}'.format(time.time() - st))

    target1, target2 = Variable(torch.rand(8,8*3*4).cuda()), Variable(torch.rand(8,8*3*4).cuda())
    loss1 = nn.MSELoss()(output[0], target1)
    loss2 = nn.MSELoss()(output[1], target2)
    loss = loss1 + loss2

    st = time.time()
    loss.backward()
    print('Time for BWD pass: {}'.format(time.time() - st))

#### Test pose mask encoder
print('==> Testing pose mask encoder <==')
M = ctrlnets.PoseMaskEncoder(num_se3=8, se3_type='se3aa', use_pivot=False,
                                 use_kinchain=False, input_channels=3, use_bn=True, nonlinearity='prelu')
M.cuda()
for k in xrange(10):
    input1 = Variable(torch.rand(32,3,240,320).cuda(), requires_grad=True)
    input2 = Variable(torch.rand(32,3,240,320).cuda(), requires_grad=True)

    st = time.time()
    output1 = M(input1)
    output2 = M(input2)
    print('Time for FWD pass: {}'.format(time.time() - st))

    target1, target2 = Variable(torch.rand(32,8,3,4).cuda()), Variable(torch.rand(32,8,240,320).cuda())

    loss1_1 = nn.MSELoss()(output1[0], target1)
    loss2_1 = nn.MSELoss()(output1[1], target2)
    loss1_2 = nn.MSELoss()(output2[0], target1)
    loss2_2 = nn.MSELoss()(output2[1], target2)
    loss  = loss1_1 + loss2_1 + loss1_2 + loss2_2

    st = time.time()
    loss.backward()
    print('Time for BWD pass: {}'.format(time.time() - st))