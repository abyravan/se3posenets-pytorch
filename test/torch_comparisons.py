# ComposeRt - Analytical grad
import torch
from torch import nn
from torch.autograd import Variable
from layers.ComposeRt import ComposeRt
torch.manual_seed(100) # seed
input  = Variable(torch.rand(2,8,3,4), requires_grad=True)
target = Variable(torch.rand(2,8,3,4))
output = ComposeRt(True)(input)
err    = nn.MSELoss()(output, target)
err.backward()
pred,grad = output.data, input.grad

########
# ComposeRtPair
import torch
from torch import nn
from torch.autograd import Variable
from layers.ComposeRtPair import ComposeRtPair
torch.manual_seed(100) # seed
input1  = Variable(torch.rand(2,8,3,4), requires_grad=True)
input2  = Variable(torch.rand(2,8,3,4), requires_grad=True)
target = Variable(torch.rand(2,8,3,4))
output = ComposeRtPair()(input1, input2)
err    = nn.MSELoss()(output, target)
err.backward()
pred,grad1,grad2 = output.data, input1.grad, input2.grad

########
# RtInverse
import torch
from torch import nn
from torch.autograd import Variable
from layers.RtInverse import RtInverse
torch.manual_seed(100) # seed
input  = Variable(torch.rand(2,8,3,4), requires_grad=True)
target = Variable(torch.rand(2,8,3,4))
output = RtInverse()(input)
err    = nn.MSELoss()(output, target)
err.backward()
pred,grad = output.data, input.grad

########
# CollapseRtPivots
import torch
from torch import nn
from torch.autograd import Variable
from layers.CollapseRtPivots import CollapseRtPivots
torch.manual_seed(100) # seed
input  = Variable(torch.rand(2,8,3,5), requires_grad=True)
target = Variable(torch.rand(2,8,3,4))
output = CollapseRtPivots()(input)
err    = nn.MSELoss()(output, target)
err.backward()
pred,grad = output.data, input.grad

########
# DepthImageToDense3DPoints
import torch
from torch import nn
from torch.autograd import Variable
from layers.DepthImageToDense3DPoints import DepthImageToDense3DPoints
torch.manual_seed(100) # seed
scale = 0.5/8
ht,wd,fy,fx,cy,cx = int(480*scale), int(640*scale), 589*scale, 589*scale, 240*scale, 320*scale
input  = Variable(torch.rand(2,1,ht,wd), requires_grad=True)
target = Variable(torch.rand(2,3,ht,wd))
output = DepthImageToDense3DPoints(ht,wd,fy,fx,cy,cx)(input)
err    = nn.MSELoss()(output, target)
err.backward()
pred,grad = output.data, input.grad

########
# NTfm3D
import torch
from torch import nn
from torch.autograd import Variable
from layers.NTfm3D import NTfm3D
torch.manual_seed(100) # seed
pts    = Variable(torch.rand(2,3,9,9), requires_grad=True)
masks  = Variable(torch.rand(2,8,9,9), requires_grad=True)
tfms   = Variable(torch.rand(2,8,3,4), requires_grad=True)
target = Variable(torch.rand(2,3,9,9))
output = NTfm3D()(pts, masks, tfms)
err    = nn.MSELoss()(output, target)
err.backward()
pred,gradpts,gradmasks,gradtfms = output.data, pts.grad, masks.grad, tfms.grad

########
# NTfm3D - CUDA
import torch
from torch import nn
from torch.autograd import Variable
from layers.NTfm3D import NTfm3D
torch.manual_seed(100) # seed
pts    = Variable(torch.rand(2,3,9,9).cuda(), requires_grad=True)
masks  = Variable(torch.rand(2,8,9,9).cuda(), requires_grad=True)
tfms   = Variable(torch.rand(2,8,3,4).cuda(), requires_grad=True)
target = Variable(torch.rand(2,3,9,9).cuda())
output = NTfm3D()(pts, masks, tfms)
err    = nn.MSELoss()(output, target)
err.backward()
pred,gradpts,gradmasks,gradtfms = output.data, pts.grad, masks.grad, tfms.grad

########
# Noise
import torch
from torch import nn
from torch.autograd import Variable
from layers.Noise import Noise
torch.manual_seed(100) # seed
max_std, slope_std, iter_count, start_iter = 0.1, 2, torch.FloatTensor([1000]), 0
input  = Variable(torch.rand(2,8,3,5), requires_grad=True)
target = Variable(torch.rand(2,8,3,5))
output = Noise(max_std, slope_std, iter_count, start_iter)(input)
err    = nn.MSELoss()(output, target)
err.backward()
pred,grad = output.data, input.grad

########
# Huber
import torch
from torch import nn
from torch.autograd import Variable
from layers.HuberLoss import HuberLoss
torch.manual_seed(100) # seed
input  = Variable(torch.rand(2,8,3,5), requires_grad=True)
target = Variable(torch.rand(2,8,3,5))
size_average, delta = True, 0.1
output = HuberLoss(size_average, delta)(input, target)
output.backward()
pred,grad = output.data, input.grad

########
# WeightedAveragePoints
import torch
from torch import nn
from torch.autograd import Variable
from layers.WeightedAveragePoints import WeightedAveragePoints
torch.manual_seed(100) # seed
pts    = Variable(torch.rand(2,3,9,9), requires_grad=True)
masks  = Variable(torch.rand(2,8,9,9), requires_grad=True)
target = Variable(torch.rand(2,8,3))
output = WeightedAveragePoints()(pts, masks)
err    = nn.MSELoss()(output, target)
err.backward()
pred,gradpts,gradmasks = output.data, pts.grad, masks.grad