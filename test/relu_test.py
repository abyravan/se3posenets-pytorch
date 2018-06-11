import torch
import torch.nn as nn
from torch.autograd import Variable
import time

# ReLU-Net
class ReLUNet(nn.Module):
    def __init__(self, dims, prelu=False):
        super(ReLUNet, self).__init__()
        self.layerlist, self.linidlist = [], []
        id = 0
        self.prelu = prelu
        for k in range(len(dims)-1):
            self.layerlist.append(nn.Linear(dims[k], dims[k+1]))
            if (k != len(dims)-2): # No ReLU right at the end
                if prelu:
                    self.layerlist.append(nn.PReLU())
                else:
                    self.layerlist.append(nn.ReLU())
            self.linidlist.append(id)
            id += 2
        self.relunet = nn.Sequential(*self.layerlist).double()

    def forward(self, x):
        self.linres = []
        for ii,model in enumerate(self.relunet):
            x = model(x)
            if ii in self.linidlist:
                self.linres.append(x)
        return x


# Setup network
dims = [64, 128, 128, 128, 64]
#dims = [16, 8, 8, 16]
net = ReLUNet(dims, prelu=True)

# Inp/Out
inp = Variable(torch.rand(1,dims[0]).double() - 0.5)

##########
# Numerical jacobian
a = time.time()
jacf  = torch.zeros(dims[0], dims[-1]).double()
outdef = net(inp).data.clone()
eps = 1e-6
for k in range(dims[0]):
    inp.data[0,k] += eps # Perturb
    jacf[:,k] = (net(inp).data.clone() - outdef) / eps
    inp.data[0,k] -= eps # Reset
print('FD: ', time.time() - a)

# ##########
# # Analytical Jacobian - ReLU
# b = time.time()
# outdef = net(inp)
# jaca1 = None
# previndexes = None
# for k in range(len(net.linidlist)):
#     id = net.linidlist[k]
#     wt = net.layerlist[id].weight
#     if (k == len(net.linidlist)-1):
#         assert jaca1 is not None, "Need to have a jac already by this point"
#         jaca1 = torch.mm(wt.data[:,previndexes], jaca1)
#     elif (k == 0):
#         assert jaca1 is None, "First time initializing the jacobians here"
#         currindexes = (net.linres[k].data > 0).squeeze().nonzero().squeeze()
#         jaca1 = wt.data[currindexes,:]
#         previndexes = currindexes
#     else:
#         assert jaca1 is not None, "Need to have a jac already by this point"
#         currindexes = (net.linres[k].data > 0).squeeze().nonzero().squeeze()
#         jaca1 = torch.mm(wt.data[currindexes,:][:,previndexes], jaca1)
#         previndexes = currindexes
# print('AN1: ', time.time()-b)

##########
# Analytical Jacobian - PReLU
b = time.time()
outdef = net(inp)
jaca = None
#previndexes = None
for k in range(len(net.linidlist)):
    id = net.linidlist[k]
    wt = net.layerlist[id].weight
    if (k == len(net.linidlist)-1):
        assert jaca is not None, "Need to have a jac already by this point"
        #jaca = torch.mm(wt.data[:,previndexes], jaca)
        jaca = torch.mm(wt.data, jaca)
    elif (k == 0):
        assert jaca is None, "Jacobians already initialized"
        currindexes = (net.linres[k].data <= 0).squeeze().nonzero().squeeze()
        jaca = wt.data.clone()
        if net.prelu:
            jaca[currindexes, :] *= net.layerlist[id+1].weight.data.view(1,-1)
        else:
            jaca[currindexes, :] *= 0
        #previndexes = currindexes
    else:
        assert jaca is not None, "Need to have a jac already by this point"
        currindexes = (net.linres[k].data <= 0).squeeze().nonzero().squeeze()
        newjac = wt.data.clone()
        if net.prelu:
            newjac[currindexes, :] *= net.layerlist[id+1].weight.data.view(1,-1)
        else:
            newjac[currindexes, :] *= 0
        jaca = torch.mm(newjac, jaca)
        #previndexes = currindexes
print('AN: ', time.time()-b)

# Find errors
diff = jacf - jaca
print(diff.abs().max(), diff.abs().min())

# # Find errors
# diff1 = jaca1 - jaca
# print(diff1.abs().max(), diff1.abs().min())