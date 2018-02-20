import torch
import scipy.optimize
import scipy.linalg
import numpy as np
import sys
import sys, os
sys.path.append("/home/barun/Projects/se3nets-pytorch/")
import data as datav
import time

import multiprocessing
#import xalglib

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
        predpts = datav.NTfm3D(pts, masks, tfms) # Transform points through non-rigid transform

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
        predpts = datav.NTfm3D(pts, masks, tfms)  # Transform points through non-rigid transform

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

def minimize(args):
    # Setup stuff
    optimclass, pts, masks, tgtpts = args
    bsz, nch, ht, wd = pts.size()
    nmsk = masks.size(1)

    # Initialize params / loss / jac fns
    initparams = torch.rand(bsz, nmsk, 3, 4).type_as(pts).view(-1).cpu().numpy()
    l = NTfm3DOptimizer()
    loss = lambda params: l.compute_loss(params, pts, masks, tgtpts)
    lossjac = lambda params: l.compute_jac(params, pts, masks, tgtpts)

    # Optimize
    res = scipy.optimize.least_squares(loss, initparams, jac=lossjac)
    return res.x

    # # Alglib
    # epsx = 0.0000000001
    # maxits = 0
    #
    # state = xalglib.minlmcreatevj(len(initparams), list(initparams))
    # xalglib.minlmsetcond(state, epsx, maxits)
    # xalglib.minlmoptimize_vj(state, loss, lossjac)
    # res, rep = xalglib.minlmresults(state)
    # print(res)
    # return res

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    ############
    import argparse
    args = argparse.Namespace()
    args.data = [
        '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions_f/',
        '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/baxter_babbling_rarm_wfixjts_5hrs_Feb10_17/postprocessmotionshalf_f/',
        '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint0/postprocessmotions_f/',
        '/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/singlejtdata/data2k/joint5/postprocessmotions_f/']
    args.img_suffix = 'sub'
    args.step_len = 2
    args.seq_len = 16
    args.train_per = 0.6
    args.val_per = 0.15
    args.ctrl_type = 'actdiffvel'
    args.batch_size = 16
    args.use_pin_memory = False
    args.num_workers = 6
    args.cuda = True
    args.se3_type = 'se3aa'
    args.pred_pivot = False
    args.num_se3 = 8
    args.se2_data = False
    args.box_data = False

    # Get default options & camera intrinsics
    args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
    args.state_labels = []
    for k in range(len(args.data)):
        load_dir = args.data[k]  # args.data.split(',,')[0]
        try:
            # Read from file
            intrinsics = datav.read_intrinsics_file(load_dir + "/intrinsics.txt")
            print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
            if args.se2_data or args.full_res:
                args.img_ht, args.img_wd = int(intrinsics['ht']), int(intrinsics['wd'])
            else:
                args.img_ht, args.img_wd = 240, 320  # All data except SE(2) data is at 240x320 resolution
            args.img_scale = 1.0 / intrinsics['s']  # Scale of the image (use directly from the data)

            # Setup camera intrinsics
            sc = float(args.img_ht) / intrinsics['ht']  # Scale factor for the intrinsics
            cam_intrinsics = {'fx': intrinsics['fx'] * sc,
                              'fy': intrinsics['fy'] * sc,
                              'cx': intrinsics['cx'] * sc,
                              'cy': intrinsics['cy'] * sc}
            print("Scale factor for the intrinsics: {}".format(sc))
        except:
            print("Could not read intrinsics file, reverting to default settings")
            args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
            cam_intrinsics = {'fx': 589.3664541825391 / 2,
                              'fy': 589.3664541825391 / 2,
                              'cx': 320.5 / 2,
                              'cy': 240.5 / 2}
        print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(args.img_ht, args.img_wd,
                                                                                    cam_intrinsics['fx'],
                                                                                    cam_intrinsics['fy'],
                                                                                    cam_intrinsics['cx'],
                                                                                    cam_intrinsics['cy']))

        # Compute intrinsic grid & add to list
        cam_intrinsics['xygrid'] = datav.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               cam_intrinsics)
        args.cam_intrinsics.append(cam_intrinsics)  # Add to list of intrinsics

        ### BOX (vs) BAXTER DATA
        # Compute extrinsics
        cam_extrinsics = datav.read_cameradata_file(load_dir + '/cameradata.txt')

        # Get dimensions of ctrl & state
        try:
            statelabels, ctrllabels, trackerlabels = datav.read_statectrllabels_file(
                load_dir + "/statectrllabels.txt")
            print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
        except:
            statelabels = datav.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
            ctrllabels = statelabels  # Just use the labels
            trackerlabels = []
            print("Could not read statectrllabels file. Reverting to labels in statelabels file")
        # args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
        # print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
        args.num_ctrl = len(ctrllabels)
        print('Num ctrl: {}'.format(args.num_ctrl))

        # Find the IDs of the controlled joints in the state vector
        # We need this if we have state dimension > ctrl dimension and
        # if we need to choose the vals in the state vector for the control
        ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
        print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

        # Add to list of intrinsics
        args.cam_extrinsics.append(cam_extrinsics)
        args.ctrl_ids.append(ctrlids_in_state)
        args.state_labels.append(statelabels)

    # Data noise
    if not hasattr(args, "add_noise_data") or (len(args.add_noise_data) == 0):
        args.add_noise_data = [False for k in range(len(args.data))]  # By default, no noise
    else:
        assert (len(args.data) == len(args.add_noise_data))
    if hasattr(args, "add_noise") and args.add_noise:  # BWDs compatibility
        args.add_noise_data = [True for k in range(len(args.data))]

    # Get mean/std deviations of dt for the data
    args.mean_dt = args.step_len * (1.0 / 30.0)
    args.std_dt = 0.005  # +- 10 ms
    print("Using default mean & std.deviation based on the step length. Mean DT: {}, Std DT: {}".format(
        args.mean_dt, args.std_dt))

    # Image suffix
    args.img_suffix = '' if (
    args.img_suffix == 'None') else args.img_suffix  # Workaround since we can't specify empty string in the yaml
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data (for baxter)
    args.baxter_labels = datav.read_statelabels_file(args.data[0] + '/statelabels.txt')
    args.mesh_ids = args.baxter_labels['meshIds']

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat',
                              'se3aar']), 'Unknown SE3 type: ' + args.se3_type
    args.delta_pivot = ''
    delta_pivot_type = ' Delta pivot type: {}'.format(args.delta_pivot) if (args.delta_pivot != '') else ''
    print('Predicting {} SE3s of type: {}.{}'.format(args.num_se3, args.se3_type, delta_pivot_type))

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    ########################
    ############ Load datasets
    # Get datasets
    load_color = None
    args.use_xyzrgb = False
    args.use_xyzhue = False
    args.reject_left_motion, args.reject_right_still = False, False
    args.add_noise = False

    print("Baxter dataset")
    valid_filter = lambda p, n, st, se, slab: datav.valid_data_filter(p, n, st, se, slab,
                                                                      mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                      reject_left_motion=args.reject_left_motion,
                                                                      reject_right_still=args.reject_right_still)
    read_seq_func = datav.read_baxter_sequence_from_disk
    ### Noise function
    # noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
    #                                                  scale_d=True, std_j=0.02) if args.add_noise else None
    noise_func = lambda d: datav.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                      defprob=0.005, noisestd=0.005)
    ### Load functions
    baxter_data = datav.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                      step_len=args.step_len, seq_len=args.seq_len,
                                                      train_per=args.train_per, val_per=args.val_per,
                                                      valid_filter=valid_filter,
                                                      cam_extrinsics=args.cam_extrinsics,
                                                      cam_intrinsics=args.cam_intrinsics,
                                                      ctrl_ids=args.ctrl_ids,
                                                      state_labels=args.state_labels,
                                                      add_noise=args.add_noise_data)
    disk_read_func = lambda d, i: read_seq_func(d, i, img_ht=args.img_ht, img_wd=args.img_wd,
                                                img_scale=args.img_scale, ctrl_type=args.ctrl_type,
                                                num_ctrl=args.num_ctrl,
                                                # num_state=args.num_state,
                                                mesh_ids=args.mesh_ids,
                                                # ctrl_ids=ctrlids_in_state,
                                                # camera_extrinsics = args.cam_extrinsics,
                                                # camera_intrinsics = args.cam_intrinsics,
                                                compute_bwdflows=True,
                                                # num_tracker=args.num_tracker,
                                                dathreshold=0.01, dawinsize=5,
                                                use_only_da=False,
                                                noise_func=noise_func,
                                                load_color=load_color,
                                                compute_normals=False,
                                                maxdepthdiff=0.1,
                                                bismooth_depths=False,
                                                bismooth_width=7,
                                                bismooth_std=0.02,
                                                supervised_seg_loss=False)  # Need BWD flows / masks if using GT masks
    train_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset = datav.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset),
                                                                       len(test_dataset)))

    ###########
    ## Sample example
    sample = train_dataset[20000]
    #torch.save(sample, 'levmartest.pth.tar')

    # Get data
    tensortype = 'torch.FloatTensor'
    if torch.cuda.is_available():
        tensortype = 'torch.cuda.FloatTensor'

    pts   = sample['points'][:-1,:,::2,::2].type(tensortype)
    masks = sample['masks'][:-1,:,::2,::2].type(tensortype) # subsample
    poses1 = sample['poses'][:-1].type(tensortype)
    poses2 = sample['poses'][1:].type(tensortype)
    deltaposes = datav.ComposeRtPair(poses2, datav.RtInverse(poses1))
    tgtpts = datav.NTfm3D(pts, masks, deltaposes)
    bsz, nch, nmsk, ht, wd = pts.size(0), pts.size(1), masks.size(1), pts.size(2), pts.size(3)
    print(bsz, nch, nmsk, ht, wd)
    print("Setup inputs, parameters, targets")
    print(deltaposes)
    # masks = torch.rand(bsz, nmsk, ht, wd).type(tensortype)
    # masks = masks/masks.sum(1).unsqueeze(1) # Normalize masks
    #
    # tfmparams_gt = torch.rand(bsz, nmsk, 3, 4).type(tensortype)  # 3x4 matrix
    # tgtpts = data.NTfm3D(pts, masks, tfmparams_gt)

    # ###########
    ## Setup stuff
    #bsz, nch, nmsk, ht, wd = 16, 3, 8, 24, 32 #120, 160
    #tensortype = 'torch.FloatTensor'
    #if torch.cuda.is_available():
    #    tensortype = 'torch.cuda.FloatTensor'
    #
    #pts = torch.rand(bsz, nch, ht, wd).type(tensortype) - 0.5
    #masks = torch.rand(bsz, nmsk, ht, wd).type(tensortype)
    #masks = masks/masks.sum(1).unsqueeze(1) # Normalize masks
    #
    #deltaposes = torch.rand(bsz, nmsk, 3, 4).type(tensortype)  # 3x4 matrix
    #tgtpts = datav.NTfm3D(pts, masks, deltaposes)
    #print("Setup inputs, parameters, targets")

    ##########
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

    ###########
    # Optimize
    nruns, mbsz = 20, 1
    import time
    tt = torch.zeros(nruns)
    for k in range(nruns):
        tti, diffmax, diffmin = [], [], [] #torch.zeros(bsz/mbsz), torch.zeros(bsz/mbsz), torch.zeros(bsz/mbsz)
        diffmax1, diffmin1 = [], []
        for j in range(0,bsz,mbsz):
            #tfmparams_init = torch.rand(mbsz,nmsk,3,4).type(tensortype).view(-1).cpu().numpy()
            tfmparams_init = torch.eye(4).view(1,1,4,4).narrow(2,0,3).expand(mbsz,nmsk,3,4).type(tensortype).view(-1).cpu().numpy()
            l = NTfm3DOptimizer()
            loss    = lambda params: l.compute_loss(params, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz))
            lossjac = lambda params: l.compute_jac( params, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz))

            st = time.time()
            res = scipy.optimize.least_squares(loss, tfmparams_init, jac=lossjac, bounds=(-1,1), max_nfev=20)
            print('Batch: {}, F:{}. J:{}'.format(j, res.nfev, res.njev))
            tti.append(time.time() - st)
            diff = res.x.reshape(mbsz,nmsk,3,4) - deltaposes.narrow(0,j,mbsz).cpu().numpy()
            diff1 = (res.x - tfmparams_init)   
            diffmax.append(diff.max())
            diffmin.append(diff.min())
            diffmax1.append(diff1.max()); diffmin1.append(diff1.min());
        tt[k] = torch.Tensor(tti).sum()
        print('Init max/min error: {:.5f}/{:.5f}, Max/min error: {:.5f}/{:.5f}, Mean/std/per example time: {:.5f}/{:.5f}/{:.5f}'.format(torch.Tensor(diffmax1).mean(),
                                                                                 torch.Tensor(diffmin1).mean(), torch.Tensor(diffmax).mean(), torch.Tensor(diffmin).mean(),
                                                                  tt[:k+1].mean(), tt[:k+1].std(), tt[:k+1].mean()/bsz))

    # ##########
    # # Optimize parallel
    # nruns, mbsz = 20, 1
    #
    # # Parallel pool
    # nthreads = 1
    # pool = multiprocessing.Pool(nthreads)
    #
    # import time
    # #from gridmap import grid_map
    # tt = torch.zeros(nruns)
    # for k in range(nruns):
    #     # Optimize & time
    #     st = time.time()
    #     args = [ (NTfm3DOptimizer, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz)) for j in range(0,bsz,mbsz) ]
    #     results = pool.map(minimize, args) # Optimize in parallel
    #
    #     # # The default queue used by grid_map is all.q. You must specify
    #     # # the `queue` keyword argument if that is not the name of your queue.
    #     # results = grid_map(minimize, args, quiet=False,
    #     #                    max_processes=nthreads, queue='all.q')
    #
    #     tt[k] = (time.time() - st)
    #
    #     # Check error w.r.t GT
    #     diff = torch.Tensor(results).view(bsz,nmsk,3,4) - tfmparams_gt
    #     print('Max/min error: {}/{}, Mean/std/per example time: {}/{}/{}'.format(diff.max(), diff.min(),
    #                                                                              tt[:k+1].mean(), tt[:k+1].std(),
    #                                                                              tt[:k+1].mean()/bsz))



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
