require 'se3depthpred'; require 'cudnn'; require 'cunn'; require 'nngraph'; 
a = torch.load('soft_se3aa_b16_poseconsis0.01/trained.data'); 


pm = a.model.posemaskmodel:float();
modules = {}
for k = 1, #(pm.modules) do; modules[k] = {}; for key, val in pairs(pm.modules[k]) do; if (type(val) == 'userdata') and (val.type ~= nil) and (val:type() == 'torch.FloatTensor') then; modules[k][key] = val; end; end; end
torch.save('posemaskmodules.t7', modules)


tm = a.model.transitionmodel:float();
modules = {}
for k = 1, #(tm.modules) do; modules[k] = {}; for key, val in pairs(tm.modules[k]) do; if (type(val) == 'userdata') and (val.type ~= nil) and (val:type() == 'torch.FloatTensor') then; modules[k][key] = val; end; end; end
torch.save('transitionmodules.t7', modules)


