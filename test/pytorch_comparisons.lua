-- ComposeRt
require 'se3depthpred';
torch.manualSeed(100); 
input = torch.rand(2,8,3,4); 
target = torch.rand(2,8,3,4); 
l = nn.ComposeRt(); 
output = l:forward(input); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward(input, graderr);

----------
-- ComposeRtPair
require 'se3depthpred';
torch.manualSeed(100); 
input1 = torch.rand(2,8,3,4); 
input2 = torch.rand(2,8,3,4); 
target = torch.rand(2,8,3,4); 
l = nn.ComposeRtPair(); 
output = l:forward({input1,input2}); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward({input1,input2}, graderr);
grad1, grad2 = unpack(grad);

----------
-- RtInverse
require 'se3depthpred';
torch.manualSeed(100); 
input = torch.rand(2,8,3,4); 
target = torch.rand(2,8,3,4); 
l = nn.RtInverse(); 
output = l:forward(input); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward(input, graderr);

----------
-- CollapseRtPivots
require 'se3depthpred';
torch.manualSeed(100); 
input = torch.rand(2,8,3,5); 
target = torch.rand(2,8,3,4); 
l = nn.CollapseRtPivots(); 
output = l:forward(input); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward(input, graderr);

----------
-- DepthImageToDensePoints3D
require 'se3depthpred';
torch.manualSeed(100); 
scale = 0.5/8
ht,wd,fy,fx,cy,cx = 480*scale, 640*scale, 589*scale, 589*scale, 240*scale, 320*scale
input = torch.rand(2,1,ht,wd); 
target = torch.rand(2,3,ht,wd); 
l = nn.DepthImageToDensePoints3D(ht,wd,fy,fx,cy,cx); 
output = l:forward(input); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward(input, graderr);

----------
-- NTfm3D
require 'se3depthpred';
torch.manualSeed(100); 
pts    = torch.rand(2,3,9,9); 
masks  = torch.rand(2,8,9,9);
tfms   = torch.rand(2,8,3,4); 
target = torch.rand(2,3,9,9); 
l = nn.NTfm3D(); 
output = l:forward({pts,masks,tfms}); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward({pts,masks,tfms}, graderr);
gradpts, gradmasks, gradtfms = unpack(grad);

----------
-- NTfm3D - CUDA
require 'se3depthpred'; require 'cunn';
torch.manualSeed(100); 
pts    = torch.rand(2,3,9,9):cuda(); 
masks  = torch.rand(2,8,9,9):cuda();
tfms   = torch.rand(2,8,3,4):cuda(); 
target = torch.rand(2,3,9,9):cuda(); 
l = nn.NTfm3D():cuda(); 
output = l:forward({pts,masks,tfms}); 
err = nn.MSECriterion():cuda(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward({pts,masks,tfms}, graderr);
gradpts, gradmasks, gradtfms = unpack(grad);

----------
-- Noise
require 'se3depthpred';
torch.manualSeed(100); 
input  = torch.rand(2,8,3,5); 
target = torch.rand(2,8,3,5); 
max_std, slope_std, iter_count, start_iter = 0.1, 2, torch.FloatTensor({1000}), 0
l = nn.Noise(max_std, iter_count, start_iter, slope_std); 
output = l:forward(input); 
err = nn.MSECriterion(); 
err:forward(output,target); 
graderr = err:backward(output,target); 
grad = l:backward(input, graderr);

----------
-- HuberCriterion
require 'se3depthpred';
torch.manualSeed(100); 
input  = torch.rand(2,8,3,5); 
target = torch.rand(2,8,3,5); 
size_average, delta = true, 0.1
l = nn.HuberCriterion(size_average, delta); 
output = l:forward(input, target);
grad = l:backward(input, target);

----------
-- WeightedAveragePoints
require 'se3depthpred';
torch.manualSeed(100);
pts    = torch.rand(2,3,9,9);
masks  = torch.rand(2,8,9,9);
target = torch.rand(2,8,3);
l = nn.WeightedAveragePoints();
output = l:forward({pts,masks});
err = nn.MSECriterion();
err:forward(output,target);
graderr = err:backward(output,target);
grad = l:backward({pts,masks}, graderr);
gradpts, gradmasks = unpack(grad);

----------
-- NormalizedMSECriterion
require 'se3depthpred';
torch.manualSeed(100);
input  = torch.randn(2,8,3,5);
target = torch.randn(2,8,3,5);
size_average, scale, defsigma = true, 0.5, 0.005
l = nn.NormalizedMSECriterion(size_average, scale, defsigma);
output = l:forward(input, target);
grad = l:backward(input, target);

----------
-- NormalizedMSESqrtCriterion
require 'se3depthpred';
torch.manualSeed(100);
input  = torch.randn(2,8,3,5);
target = torch.randn(2,8,3,5);
size_average, scale, defsigma = true, 0.5, 0.005
l = nn.NormalizedMSESqrtCriterion(size_average, scale, defsigma);
output = l:forward(input, target);
grad = l:backward(input, target);

----------
-- SE3ToRt
require 'nn';
dofile('SE3ToRt.lua')
torch.manualSeed(100);
bsz, nse3, se3_type, has_pivot = 2, 2, 'se3quat', true
ncols  = has_pivot and 5 or 4
npivot = has_pivot and 3 or 0
ndim   = ((se3_type == 'se3quat') and  7) or
         ((se3_type == 'affine')  and 12) or 6;
input  = torch.rand(bsz, nse3, ndim+npivot)
target = torch.rand(bsz, nse3, 3, ncols);
l = nn.SE3ToRt(se3_type, has_pivot);
pred = l:forward(input);
err  = nn.MSECriterion();
err:forward(pred,target);
graderr = err:backward(pred,target);
grad = l:backward(input, graderr);