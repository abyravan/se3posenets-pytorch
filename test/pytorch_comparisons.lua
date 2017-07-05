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

