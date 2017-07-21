import ctrlnetworks
import torch
import time
from torch.autograd import Variable
import torch.nn as nn

#### Test transition model
print('==> Testing transition model <==')
T = ctrlnetworks.TransitionModel(num_ctrl=7, num_se3=8, se3_type='se3aa', use_pivot=False,
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
M = ctrlnetworks.PoseMaskEncoder(num_se3=8, se3_type='se3aa', use_pivot=False,
                                 use_kinchain=False, input_channels=3, use_bn=True, nonlinearity='prelu')
M.cuda()
for k in xrange(10):
    input1 = Variable(torch.rand(32,3,240,320).cuda(), requires_grad=True)
    input2 = Variable(torch.rand(32,3,240,320).cuda(), requires_grad=True)

    st = time.time()
    output1 = M(input1)
    output2 = M(input2)
    print('Time for FWD pass: {}'.format(time.time() - st))

    target1, target2 = Variable(torch.rand(32,8,240,320).cuda()), Variable(torch.rand(32,8,3,4).cuda())
    loss1_1 = nn.MSELoss()(output1[0], target1)
    loss2_1 = nn.MSELoss()(output1[1], target2)
    loss1_2 = nn.MSELoss()(output2[0], target1)
    loss2_2 = nn.MSELoss()(output2[1], target2)
    loss  = loss1_1 + loss2_1 + loss1_2 + loss2_2

    st = time.time()
    loss.backward()
    print('Time for BWD pass: {}'.format(time.time() - st))