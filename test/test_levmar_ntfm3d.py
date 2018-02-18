import torch
import scipy.optimize
import numpy as np
import sys
import sys, os
sys.path.append("/home/barun/Projects/se3nets-pytorch/")
import data
import time

class NTfm3DOptimizer:
    def __init__(self):
        #super(NTfm3DOptimizer, self)
        self.jac = None
        #self.ntfm3d = data.NTfm3D

    def compute_loss(self, tfmparams, pts, masks, targets):
        # Setup loss computations
        bsz, nch, ht, wd = pts.size()
        nmaskch = masks.size(1)
        assert targets.is_same_size(pts), "Input/Output pts need to be of same size"
        assert masks.size() == torch.Size([bsz, nmaskch, ht, wd]), "Tfms need to be of size [bsz x nch x 3 x 4]"

        # Compute loss
        tfms = torch.from_numpy(tfmparams).view(bsz,nmaskch,3,4).type_as(pts).clone() # 3 x 4 matrix of params
        predpts = data.NTfm3D(pts, masks, tfms) # Transform points through non-rigid transform

        # Compute residual & loss
        residual = (predpts - targets) # B x 3 x H x W
        loss = torch.pow(residual, 2).sum(1).view(-1).cpu().numpy() # "BHW" vector of losses
        return loss

    def compute_jac(self, tfmparams, pts, masks, targets):
        # Setup loss computations
        bsz, nch, ht, wd = pts.size()
        nmaskch = masks.size(1)
        assert targets.is_same_size(pts), "Input/Output pts need to be of same size"
        assert masks.size() == torch.Size([bsz, nmaskch, ht, wd]), "Tfms need to be of size [bsz x nch x 3 x 4]"

        # Compute loss
        tfms = torch.from_numpy(tfmparams).view(bsz, nmaskch, 3, 4).type_as(pts)  # 3 x 4 matrix of params
        predpts = data.NTfm3D(pts, masks, tfms)  # Transform points through non-rigid transform

        # Compute gradient of residual
        gradresidual = 2*(predpts - targets) # B x 3 x H x W

        # # Output jacobial is dl/dp (l = loss, p = params)
        # if self.jac is None:
        #     self.jac = torch.zeros(bsz, ht, wd, nmaskch, 3, 4).type_as(pts) # num_pts x num_params (across all batches)
        #
        # # Compute jac w.r.t translation parameters
        # gxtm, gytm, gztm = gradresidual.narrow(1,0,1) * masks, \
        #                    gradresidual.narrow(1,1,1) * masks, \
        #                    gradresidual.narrow(1,2,1) * masks # B x k x H x W (t1, t2, t3)
        # self.jac[:,:,:,:,0,3] = gxtm.permute(0,2,3,1) # B x H x W x k (t1)
        # self.jac[:,:,:,:,1,3] = gytm.permute(0,2,3,1) # B x H x W x k (t2)
        # self.jac[:,:,:,:,2,3] = gztm.permute(0,2,3,1) # B x H x W x k (t3)

        # Output jacobial is dl/dp (l = loss, p = params)
        if self.jac is None:
            self.jac = torch.zeros(bsz*bsz, nmaskch, ht, wd, 3, 4).type_as(pts)  # num_pts x num_params (across all batches)

        # Compute jac w.r.t translation parameters
        self.jac[::(bsz+1), :, :, :, 0, 3] = gradresidual.narrow(1, 0, 1) * masks  # B x K x H x W (t1) (gxt * m)
        self.jac[::(bsz+1), :, :, :, 1, 3] = gradresidual.narrow(1, 1, 1) * masks  # B x K x H x W (t2) (gyt * m)
        self.jac[::(bsz+1), :, :, :, 2, 3] = gradresidual.narrow(1, 2, 1) * masks  # B x K x H x W (t3) (gzt * m)
        gxtm, gytm, gztm = self.jac[::(bsz+1), :, :, :, 0, 3], \
                           self.jac[::(bsz+1), :, :, :, 1, 3], \
                           self.jac[::(bsz+1), :, :, :, 2, 3]

        # Compute jac w.r.t rotation parameters (r00, r10, r20)
        self.jac[::(bsz+1), :, :, :, 0, 0] = gxtm * pts.narrow(1, 0, 1) # (gxt * x * m)
        self.jac[::(bsz+1), :, :, :, 1, 0] = gytm * pts.narrow(1, 0, 1) # (gyt * x * m)
        self.jac[::(bsz+1), :, :, :, 2, 0] = gztm * pts.narrow(1, 0, 1) # (gzt * x * m)

        # Compute jac w.r.t rotation parameters (r01, r11, r21)
        self.jac[::(bsz+1), :, :, :, 0, 1] = gxtm * pts.narrow(1, 1, 1) # (gxt * y * m)
        self.jac[::(bsz+1), :, :, :, 1, 1] = gytm * pts.narrow(1, 1, 1) # (gyt * y * m)
        self.jac[::(bsz+1), :, :, :, 2, 1] = gztm * pts.narrow(1, 1, 1) # (gzt * y * m)

        # Compute jac w.r.t rotation parameters (r01, r11, r21)
        self.jac[::(bsz+1), :, :, :, 0, 2] = gxtm * pts.narrow(1, 2, 1) # (gxt * z * m)
        self.jac[::(bsz+1), :, :, :, 1, 2] = gytm * pts.narrow(1, 2, 1) # (gyt * z * m)
        self.jac[::(bsz+1), :, :, :, 2, 2] = gztm * pts.narrow(1, 2, 1) # (gzt * z * m)

        return self.jac.view(bsz,bsz,nmaskch,ht,wd,3,4).permute(0,3,4,1,2,5,6).clone().view(bsz*ht*wd, bsz*nmaskch*3*4).cpu().numpy()

#
# Setup stuff
bsz, nch, nmsk, ht, wd = 16, 3, 8, 120, 160
tensortype = 'torch.FloatTensor'
if torch.cuda.is_available():
    tensortype = 'torch.cuda.FloatTensor'

pts = torch.rand(bsz, nch, ht, wd).type(tensortype) - 0.5
masks = torch.rand(bsz, nmsk, ht, wd).type(tensortype)
masks = masks/masks.sum(1).unsqueeze(1) # Normalize masks

tfmparams_gt = torch.rand(bsz, nmsk, 3, 4).type(tensortype)  # 3x4 matrix
tgtpts = data.NTfm3D(pts, masks, tfmparams_gt)
print("Setup inputs, parameters, targets")

# ### Finite difference to check stuff
# params_t = torch.rand(bsz, nmsk, 3, 4).double().view(-1).numpy()
# l1 = NTfm3DOptimizer()
# lossb = l1.compute_loss(params_t, pts, masks, tgtpts)
# jacb  = l1.compute_jac(params_t, pts, masks, tgtpts)
# jacf  = torch.from_numpy(jacb).clone().zero_().numpy()
# eps = 1e-6
# for k in range(len(params_t)):
#     params_t[k] += eps # Perturb
#     lossf = l1.compute_loss(params_t, pts, masks, tgtpts)
#     jacf[:,k] = (lossf - lossb) / eps
#     params_t[k] -= eps # Reset
# diff = jacf - jacb
# print(np.abs(diff).max(), np.abs(diff).min())

# Optimize
nruns, mbsz = 20, 1
import time
tt = torch.zeros(nruns)
for k in range(nruns):
    tti, diffmax, diffmin = [], [], [] #torch.zeros(bsz/mbsz), torch.zeros(bsz/mbsz), torch.zeros(bsz/mbsz)
    for j in range(0,bsz,mbsz):
        tfmparams_init = torch.rand(mbsz,nmsk,3,4).type(tensortype).view(-1).cpu().numpy()
        l = NTfm3DOptimizer()
        loss    = lambda params: l.compute_loss(params, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz))
        lossjac = lambda params: l.compute_jac( params, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz))

        st = time.time()
        res = scipy.optimize.least_squares(loss, tfmparams_init, jac=lossjac)
        tti.append(time.time() - st)
        diff = res.x.reshape(mbsz,nmsk,3,4) - tfmparams_gt.narrow(0,j,mbsz).cpu().numpy()
        diffmax.append(diff.max())
        diffmin.append(diff.min())
    tt[k] = torch.Tensor(tti).sum()
    print('Max/min error: {}/{}, Mean/std/per example time: {}/{}/{}'.format(torch.Tensor(diffmax).mean(),
                                                                             torch.Tensor(diffmin).mean(),
                                                              tt[:k+1].mean(), tt[:k+1].std(), tt[:k+1].mean()/bsz))


'''
###########################
class LossVal:
    def __init__(self):
        super(LossVal, self)

    def compute_loss(self, params, x, y):
        self.x, self.y = x.view(3,-1), y.view(3,-1)
        self.tfm = torch.from_numpy(params).view(3,4).type_as(x) # 3x4 matrix
        R, t = self.tfm[:3,:3], self.tfm[:,3:] # 3x3 matrix, 3x1 matrix
        yp = torch.mm(R, self.x) + t
        self.res = (yp - self.y)
        loss = torch.pow(self.res, 2).sum(0).view(-1).numpy() # "N" vector of losses
        return loss

    def compute_jac(self, params, x, y):
        # First compute loss (for now)
        self.x, self.y = x.view(3, -1), y.view(3, -1)
        self.tfm = torch.from_numpy(params).view(3, 4).type_as(x)  # 3x4 matrix
        R, t = self.tfm[:3, :3], self.tfm[:, 3:]  # 3x3 matrix, 3x1 matrix
        yp = torch.mm(R, self.x) + t
        res2 = 2*(yp - self.y)

        # Output jac is dl/dp (l = loss, p = params)
        dldp  = torch.zeros(self.x.size(1), self.tfm.nelement()).type_as(x) # num_pts x num_params
        dldt  = res2 # 3 x num_pts
        dldR1  = self.x * res2.narrow(0,0,1) # (3 x num_pts) * (1 x num_pts)
        dldR2  = self.x * res2.narrow(0,1,1) # (3 x num_pts) * (1 x num_pts)
        dldR3  = self.x * res2.narrow(0,2,1) # (3 x num_pts) * (1 x num_pts)
        dldp[:,[0,1,2]]  = dldR1.t() # r11, r12, r13
        dldp[:,[4,5,6]]  = dldR2.t() # r11, r12, r13
        dldp[:,[8,9,10]] = dldR3.t() # r11, r12, r13
        dldp[:,[3,7,11]] = dldt.t() # t1, t2, t3
        return dldp.numpy()

l = LossVal()
loss = l.compute_loss(tfmparams_gt.numpy(), pts, tgtpts)
lossjac = l.compute_jac(tfmparams_gt.numpy(), pts, tgtpts)

l1 = NTfm3DOptimizer()
loss1 = l1.compute_loss(tfmparams_gt.numpy(), pts, masks, tgtpts)
lossjac1 = l1.compute_jac(tfmparams_gt.numpy(), pts, masks, tgtpts)

# Setup stuff
x = torch.rand(3,100) - 0.5
params_gt = torch.rand(3,4) # 3x4 matrix
R_gt, t_gt = params_gt[:3,:3], params_gt[:,3:] # 3x3 matrix, 3x1 matrix
y = torch.mm(R_gt, x) + t_gt

# ### Finite difference to check stuff
# params_t = torch.rand(12).double().numpy()
# l1 = LossVal()
# lossb = l1.compute_loss(params_t, x, y)
# jacb  = l1.compute_jac(params_t, x, y)
# jacf  = torch.from_numpy(jacb).clone().zero_().numpy()
# eps = 1e-6
# for k in range(len(params_t)):
#     params_t[k] += eps # Perturb
#     lossf = l1.compute_loss(params_t, x, y)
#     jacf[:,k] = (lossf - lossb) / eps
#     params_t[k] -= eps # Reset
# diff = jacf - jacb
# print(np.abs(diff).max(), np.abs(diff).min())

# Optimize
params_init = torch.rand(3,4).view(-1).numpy()
l = LossVal()
loss    = lambda params: l.compute_loss(params, x, y)
lossjac = lambda params: l.compute_jac(params, x, y)

res = scipy.optimize.least_squares(loss, params_init, jac=lossjac)
'''
