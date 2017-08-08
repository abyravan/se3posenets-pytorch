require 'se3depthpred'; require 'cudnn'; require 'cunn'; require 'nngraph';

-- Load network / data
data = torch.load('test/torchctrlnet/soft_se3aa_b16_poseconsis0.01/trained.data');
posemasknet   = data.model.posemaskmodel
transitionnet = data.model.transitionmodel
posemasknet:evaluate()
transitionnet:evaluate()

-- Sample data
torch.manualSeed(100); 
ptclouds = torch.rand(2,3,240,320):cuda()
ctrls    = torch.rand(2,7):cuda()
print('Ptclouds: ' .. ptclouds:max() ..' '.. ptclouds:min() ..' '.. ptclouds:clone():abs():mean())
print('Ctrls: ' .. ctrls:max() ..' '.. ctrls:min() ..' '.. ctrls:clone():abs():mean())

poses, masks = unpack(posemasknet:forward(ptclouds))
nextposes    = transitionnet:forward({poses, ctrls})
print('Poses: ' .. poses:max() ..' '.. poses:min() ..' '.. poses:clone():abs():mean())
print('Masks: ' .. masks:max() ..' '.. masks:min() ..' '.. masks:clone():abs():mean())
print('Next: ' .. nextposes:max() ..' '.. nextposes:min() ..' '.. nextposes:clone():abs():mean())


