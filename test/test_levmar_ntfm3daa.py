import torch
import sys; sys.path.append('/home/barun/Projects/se3nets-pytorch/')
import data
import time
import scipy.optimize
import numpy as np

#### General helpers
# Create a skew-symmetric matrix "S" of size [B x 3 x 3] (passed in) given a [B x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    output = torch.zeros(N,3,3).type_as(vector)
    output[:, 0, 1] = -vector[:, 2]
    output[:, 1, 0] =  vector[:, 2]
    output[:, 0, 2] =  vector[:, 1]
    output[:, 2, 0] = -vector[:, 1]
    output[:, 1, 2] = -vector[:, 0]
    output[:, 2, 1] =  vector[:, 0]
    return output

# Compute the rotation matrix R from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def create_rot_from_aa(params):
    # Get the un-normalized axis and angle
    N, eps = params.size(0), 1e-12
    axis = params.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle
    small = (angle2 < eps)

    # Create Identity matrix
    I = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(params)

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute A = (sin(theta)/theta)
    A = torch.sin(angle) / angle
    A[small] = 1.0 # sin(0)/0 ~= 1

    # Compute B = (1 - cos(theta)/theta^2)
    B = (1 - torch.cos(angle)) / angle2
    B[small] = 1/2 # lim 0-> 0 (1 - cos(0))/0^2 = 1/2

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    R = I + K * A.expand(N, 3, 3) + K2 * B.expand(N, 3, 3)
    return R

# Compute the rotation matrix R & translation vector from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def AAToRt(params):
    # Check dimensions
    bsz, nse3, ndim = params.size()
    N = bsz*nse3
    assert (ndim == 6)

    # Trans | Rot params
    params_v = params.view(N, ndim, 1).clone()
    rotparam   = params_v.narrow(1,3,3) # so(3)
    transparam = params_v.narrow(1,0,3) # R^3

    # Compute rotation matrix (Bk x 3 x 3)
    R = create_rot_from_aa(rotparam)

    # Final tfm
    return torch.cat([R, transparam], 2).view(bsz,nse3,3,4).clone() # B x K x 3 x 4

# Compute the jacobians of the 3x4 transform matrix w.r.t transform parameters
def AAToRtJac(params):
    # Check dimensions
    bsz, nse3, ndim = params.size()
    N = bsz * nse3
    eps = 1e-12
    assert (ndim == 6)

    # Create jacobian matrix
    J = torch.zeros(bsz*bsz, nse3*nse3, 3, 4, ndim).type_as(params)
    J[::(bsz+1), ::(nse3+1), 0, 3, 0] = 1 # Translation vector passed out as is (first 3 params are translations)
    J[::(bsz+1), ::(nse3+1), 1, 3, 1] = 1 # Translation vector passed out as is (first 3 params are translations)
    J[::(bsz+1), ::(nse3+1), 2, 3, 2] = 1 # Translation vector passed out as is (first 3 params are translations)

    # Trans | Rot params
    params_v = params.view(N, ndim, 1).clone()
    rotparam = params_v.narrow(1, 3, 3)  # so(3)

    # Get the un-normalized axis and angle
    axis = rotparam.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    small = (angle2 < eps)  # Don't need gradient w.r.t this operation (also because of pytorch error: https://discuss.pytorch.org/t/get-error-message-maskedfill-cant-differentiate-the-mask/9129/4)

    # Compute rotation matrix
    R = create_rot_from_aa(rotparam)

    # Create Identity matrix
    I = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).type_as(params)

    ######## Jacobian computation
    # Compute: v x (Id - R) for all the columns of (Id-R)
    vI = torch.cross(axis.expand_as(I), (I - R), 1)  # (Bk) x 3 x 3 => v x (Id - R)

    # Compute [v * v' + v x (Id - R)] / ||v||^2
    vV = torch.bmm(axis, axis.transpose(1, 2))  # (Bk) x 3 x 3 => v * v'
    vV = (vV + vI) / (angle2.view(N, 1, 1).expand_as(vV))  # (Bk) x 3 x 3 => [v * v' + v x (Id - R)] / ||v||^2

    # Iterate over the 3-axis angle parameters to compute their gradients
    # ([v * v' + v x (Id - R)] / ||v||^2 _ k) x (R) .* gradOutput  where "x" is the cross product
    for k in range(3):
        # Create skew symmetric matrix
        skewsym = create_skew_symmetric_matrix(vV.narrow(2, k, 1))

        # For those AAs with angle^2 < threshold, gradient is different
        # We assume angle = 0 for these AAs and update the skew-symmetric matrix to be one w.r.t identity
        if (small.any()):
            vec = torch.zeros(1, 3).type_as(skewsym)
            vec[0,k] = 1  # Unit vector
            idskewsym = create_skew_symmetric_matrix(vec)
            for i in range(N):
                if (angle2[i].squeeze()[0] < eps):
                    skewsym[i].copy_(idskewsym)  # Use the new skew sym matrix (around identity)

        # Compute the jacobian elements now
        J[::(bsz+1), ::(nse3+1), :, 0:3, k+3] = torch.bmm(skewsym, R) # (Bk) x 3 x 3

    return J.view(bsz, bsz, nse3, nse3, 12, ndim).permute(0,2,4,1,3,5).clone().view(bsz*nse3*12, bsz*nse3*ndim).clone()

# Non-rigid transform using Axis-Angle params
class NTfm3DAAOptimizer:
    def __init__(self):
        self.J = None
        self.Jfull = None

    def compute_loss(self, params_n, pts, masks, tgtpts):
        # Setup loss computations
        bsz, nch, ht, wd = pts.size()
        nse3 = masks.size(1)
        assert tgtpts.is_same_size(pts), "Input/Output pts need to be of same size"
        assert masks.size() == torch.Size([bsz, nse3, ht, wd]), "Masks incorrect size"

        # Compute R/t from axis-angle params
        params = torch.from_numpy(params_n).view(bsz, nse3, 6).type_as(pts).clone() # B x K x 6 params
        tfms = AAToRt(params) # B x K x 3 x 4

        # Transform points
        tfmpts = data.NTfm3D(pts, masks, tfms)

        # Compute error to target points
        residual = (tfmpts - tgtpts)  # B x 3 x H x W
        loss = torch.pow(residual, 2).sum(1).view(-1).cpu().numpy()  # "BHW" vector of losses
        return loss

    # Compute Jacobian for Non-rigid transform using Axis-Angle params
    def compute_jac(self, params_n, pts, masks, tgtpts):
        # Setup loss computations
        bsz, nch, ht, wd = pts.size()
        nse3, eps = masks.size(1), 1e-12
        assert tgtpts.is_same_size(pts), "Input/Output pts need to be of same size"
        assert masks.size() == torch.Size([bsz, nse3, ht, wd]), "Masks incorrect size"

        # Compute R/t from axis-angle params
        params = torch.from_numpy(params_n).view(bsz, nse3, 6).type_as(pts).clone()  # B x K x 6 params

        ######## FWD pass
        # Compute R/t from axis-angle params
        tfms = AAToRt(params)  # B x K x 3 x 4

        # Transform points
        tfmpts = data.NTfm3D(pts, masks, tfms)

        # Compute gradient of error
        graderror = 2 * (tfmpts - tgtpts)  # B x 3 x H x W

        ######### Compute jacobian of loss w.r.t R/t matrix first
        # Create jacobian matrix
        if self.J is None:
            self.J     = torch.zeros(bsz * bsz, nse3, ht, wd, 3, 4).type_as(params)
            self.Jfull = torch.zeros(bsz * ht * wd, bsz * nse3 * 6).type_as(params)

        # Compute jacobian w.r.t translation
        self.J[::(bsz + 1), :, :, :, 0, 3] = graderror.narrow(1, 0, 1) * masks  # B x K x H x W (t1) (gxt * m)
        self.J[::(bsz + 1), :, :, :, 1, 3] = graderror.narrow(1, 1, 1) * masks  # B x K x H x W (t2) (gyt * m)
        self.J[::(bsz + 1), :, :, :, 2, 3] = graderror.narrow(1, 2, 1) * masks  # B x K x H x W (t3) (gzt * m)
        gxtm, gytm, gztm = self.J[::(bsz + 1), :, :, :, 0, 3], \
                           self.J[::(bsz + 1), :, :, :, 1, 3], \
                           self.J[::(bsz + 1), :, :, :, 2, 3]

        # Compute jac w.r.t rotation parameters (r00, r10, r20)
        self.J[::(bsz + 1), :, :, :, 0, 0] = gxtm * pts.narrow(1, 0, 1)  # (gxt * x * m)
        self.J[::(bsz + 1), :, :, :, 1, 0] = gytm * pts.narrow(1, 0, 1)  # (gyt * x * m)
        self.J[::(bsz + 1), :, :, :, 2, 0] = gztm * pts.narrow(1, 0, 1)  # (gzt * x * m)

        # Compute jac w.r.t rotation parameters (r01, r11, r21)
        self.J[::(bsz + 1), :, :, :, 0, 1] = gxtm * pts.narrow(1, 1, 1)  # (gxt * y * m)
        self.J[::(bsz + 1), :, :, :, 1, 1] = gytm * pts.narrow(1, 1, 1)  # (gyt * y * m)
        self.J[::(bsz + 1), :, :, :, 2, 1] = gztm * pts.narrow(1, 1, 1)  # (gzt * y * m)

        # Compute jac w.r.t rotation parameters (r01, r11, r21)
        self.J[::(bsz + 1), :, :, :, 0, 2] = gxtm * pts.narrow(1, 2, 1)  # (gxt * z * m)
        self.J[::(bsz + 1), :, :, :, 1, 2] = gytm * pts.narrow(1, 2, 1)  # (gyt * z * m)
        self.J[::(bsz + 1), :, :, :, 2, 2] = gztm * pts.narrow(1, 2, 1)  # (gzt * z * m)

        # Reshape Jac
        J_L_T = self.J.view(bsz,bsz,nse3,ht,wd,3,4).permute(0,3,4,1,2,5,6).clone().view(bsz*ht*wd, bsz*nse3*3*4) # (BHW) x (BK12)

        ######### Compute jacobian of R/t matrix w.r.t AA params next
        # Create jacobian matrix
        J_T_AA = AAToRtJac(params) # (BK12) x (BK6)

        ######### Compute jacobian of loss w.r.t AA params (chain rule: J_L_AA = J_L_T * J_T_AA)
        torch.mm(J_L_T, J_T_AA, out=self.Jfull)
        return self.Jfull.cpu().numpy()

# # ###########
# # Setup stuff
# bsz, nch, nmsk, ht, wd = 16, 3, 8, 24, 32 #120, 160
# tensortype = 'torch.DoubleTensor'
# if torch.cuda.is_available():
#    tensortype = 'torch.cuda.FloatTensor'
#
# pts   = torch.rand(bsz, nch, ht, wd).type(tensortype) - 0.5
# masks = torch.rand(bsz, nmsk, ht, wd).type(tensortype)
# masks = masks/masks.sum(1).unsqueeze(1) # Normalize masks
#
# params = torch.rand(bsz, nmsk, 6).type(tensortype)  # 3x4 matrix
# tfms   = AAToRt(params) # B x K x 3 x 4
# tgtpts = data.NTfm3D(pts, masks, tfms)
# print("Setup inputs, parameters, targets")

# ###########
# Sim data
tensortype = 'torch.FloatTensor'
if torch.cuda.is_available():
    tensortype = 'torch.cuda.FloatTensor'

############
load_from_tar = int(sys.argv[1])
if load_from_tar:
    print("Loading from tar file")
    sample = torch.load('/home/barun/Projects/se3nets-pytorch/levmartest.pth.tar')
else:
    print("Loading from disk")
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
            intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
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
        cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               cam_intrinsics)
        args.cam_intrinsics.append(cam_intrinsics)  # Add to list of intrinsics

        ### BOX (vs) BAXTER DATA
        # Compute extrinsics
        cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

        # Get dimensions of ctrl & state
        try:
            statelabels, ctrllabels, trackerlabels = data.read_statectrllabels_file(
                load_dir + "/statectrllabels.txt")
            print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
        except:
            statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
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
    args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
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
    valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                                      mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                      reject_left_motion=args.reject_left_motion,
                                                                      reject_right_still=args.reject_right_still)
    read_seq_func = data.read_baxter_sequence_from_disk
    ### Noise function
    # noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
    #                                                  scale_d=True, std_j=0.02) if args.add_noise else None
    noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                      defprob=0.005, noisestd=0.005)
    ### Load functions
    baxter_data = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
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
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset),
                                                                       len(test_dataset)))

    # Sample example
    id = np.random.randint(0,len(train_dataset))
    print("Example ID: {}".format(id))
    sample = train_dataset[id]
    #torch.save(sample, 'levmartest.pth.tar')

pts   = sample['points'][:-1,:,::2,::2].type(tensortype)
masks = sample['masks'][:-1,:,::2,::2].type(tensortype) # subsample
poses1 = sample['poses'][:-1].type(tensortype)
poses2 = sample['poses'][1:].type(tensortype)
tfms = data.ComposeRtPair(poses2, data.RtInverse(poses1))
# tfms[:,:,:,0:3] = torch.eye(3).view(1,1,3,3).expand(pts.size(0), masks.size(1), 3, 3)
tgtpts = data.NTfm3D(pts, masks, tfms)
bsz, nch, nmsk, ht, wd = pts.size(0), pts.size(1), masks.size(1), pts.size(2), pts.size(3)
print(bsz, nch, nmsk, ht, wd)
print("Setup inputs, parameters, targets")

# #########
# ### Finite difference to check stuff
# import numpy as np
# params_t = torch.rand(bsz, nmsk, 6).type(tensortype).view(-1).numpy()
# lossb = NTfm3D_AA(params_t, pts, masks, tgtpts)
# jacb  = NTfm3D_AA_Jac(params_t, pts, masks, tgtpts)
# jacf  = torch.from_numpy(jacb).clone().zero_().numpy()
# eps = 1e-6
# for k in range(len(params_t)):
#     params_t[k] += eps # Perturb
#     lossf = NTfm3D_AA(params_t, pts, masks, tgtpts)
#     jacf[:,k] = (lossf - lossb) / eps
#     params_t[k] -= eps # Reset
# diff = jacf - jacb
# print(np.abs(diff).max(), np.abs(diff).min())

###########
# Optimize
nruns, mbsz = int(sys.argv[2]), 1
init_type = str(sys.argv[3])
print("Num runs: {}, Init type: {}".format(nruns, init_type))
tt = torch.zeros(nruns)
for k in range(nruns):
    tti, diffmax, diffmin = [], [], [] #torch.zeros(bsz/mbsz), torch.zeros(bsz/mbsz), torch.zeros(bsz/mbsz)
    diffmax1, diffmin1 = [], []
    for j in range(0,bsz,mbsz):
        if init_type == 'rand':
            tfmparams_init = torch.rand(mbsz,nmsk,6).type(tensortype).view(-1).cpu().numpy()
        elif init_type == 'randsmall':
            tfmparams_init = torch.zeros(mbsz,nmsk,6).uniform_(-0.01,0.01).type(tensortype).view(-1).cpu().numpy()
        elif init_type == 'zero':
            tfmparams_init = torch.zeros(mbsz,nmsk,6).type(tensortype).view(-1).cpu().numpy()
        else:
            assert(False)
        l = NTfm3DAAOptimizer()
        loss    = lambda params: l.compute_loss(params, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz))
        lossjac = lambda params: l.compute_jac(params, pts.narrow(0,j,mbsz), masks.narrow(0,j,mbsz), tgtpts.narrow(0,j,mbsz))

        st = time.time()
        res = scipy.optimize.least_squares(loss, tfmparams_init, jac=lossjac)
        tti.append(time.time() - st)
        print('Test: {}/{}, Example: {}/{}, F:{}. J:{}'.format(k+1, nruns, j+1, pts.size(0), res.nfev, res.njev))
        optimtfms = AAToRt(torch.from_numpy(res.x).view(mbsz,nmsk,6).type_as(tfms))
        inittfms  = AAToRt(torch.from_numpy(tfmparams_init).view(mbsz,nmsk,6).type_as(tfms))
        diff  = optimtfms - tfms.narrow(0,j,mbsz).cpu()
        diff1 = inittfms  - tfms.narrow(0,j,mbsz).cpu()
        #diff = res.x.reshape(mbsz,nmsk,6) - params.narrow(0,j,mbsz).cpu().numpy()
        #diff1 = (res.x - tfmparams_init)
        diffmax.append(diff.max()); diffmin.append(diff.min())
        diffmax1.append(diff1.max()); diffmin1.append(diff1.min());
    tt[k] = torch.Tensor(tti).sum()
    print('Init max/min error: {:.5f}/{:.5f}, Max/min error: {:.5f}/{:.5f}, Mean/std/per example time: {:.5f}/{:.5f}/{:.5f}'.format(torch.Tensor(diffmax1).mean(),
                                                              torch.Tensor(diffmin1).mean(), torch.Tensor(diffmax).mean(), torch.Tensor(diffmin).mean(),
                                                              tt[:k+1].mean(), tt[:k+1].std(), tt[:k+1].mean()/bsz))

# #########
# ### Finite difference to check stuff
# from torch.autograd import Variable
# def loss1(params_n, pts, masks, tgtpts):
#     # Setup loss computations
#     bsz, nch, ht, wd = pts.size()
#     nse3, eps = masks.size(1), 1e-12
#     assert tgtpts.is_same_size(pts), "Input/Output pts need to be of same size"
#     assert masks.size() == torch.Size([bsz, nse3, ht, wd]), "Masks incorrect size"
#
#     # Compute R/t from axis-angle params
#     params = torch.from_numpy(params_n).view(bsz, nse3, 6).type_as(pts).clone()  # B x K x 6 params
#     tfms = se3nn.SE3ToRt('se3aa')(Variable(params))
#     tfmpts = se3nn.NTfm3D()(Variable(pts), Variable(masks), tfms)
#     residual = (tfmpts.data - tgtpts)  # B x 3 x H x W
#     loss = torch.pow(residual, 2).sum(1).view(-1).cpu().numpy()  # "BHW" vector of losses
#     return loss
#
#
# import numpy as np
# import se3layers as se3nn
# params_t = torch.rand(bsz, nmsk, 6).type(tensortype).view(-1).numpy()
# lossb = loss1(params_t, pts, masks, tgtpts)
# jacb  = NTfm3D_AA_Jac(params_t, pts, masks, tgtpts)
# jacf  = torch.from_numpy(jacb).clone().zero_().numpy()
# eps = 1e-6
# for k in range(len(params_t)):
#     params_t[k] += eps # Perturb
#     lossf = loss1(params_t, pts, masks, tgtpts)
#     jacf[:,k] = (lossf - lossb) / eps
#     params_t[k] -= eps # Reset
# diff = jacf - jacb
# print(np.abs(diff).max(), np.abs(diff).min())

#########################################################33
# Test for only AA_TO_RT part

# # ###########
# ## Setup stuff
# bsz, nse3 = 2, 3
# tensortype = 'torch.DoubleTensor'
# params = torch.rand(bsz, nse3, 6).type(tensortype) # AA params
# initparams = params.clone()
#
# ##########
# ### Finite difference to check stuff
# tfmb  = AAToRt(params).clone()
# jacb  = AAToRtJac(params)
# jacf  = jacb.clone().zero_()
# params_v = params.view(-1)
# eps = 1e-6
# for k in range(len(params_v)):
#     params_v[k] += eps # Perturb
#     tfmf = AAToRt(params.clone()).clone()
#     jacf[:,k] = (tfmf - tfmb) / eps
#     params_v[k] -= eps # Reset
# diff = jacf - jacb
# print('Diff: ', diff.max(), diff.min())

# ### Finite difference to check stuff
# import se3layers as se3nn
# tfmb_1  = se3nn.SE3ToRt('se3aa')(torch.autograd.Variable(params))
# jacb_1  = AAToRtJac(params)
# jacf_1  = jacb_1.clone().zero_()
# params_v_1 = params.view(-1)
# eps = 1e-6
# for k in range(len(params_v_1)):
#     params_v_1[k] += eps # Perturb
#     tfmf_1 = se3nn.SE3ToRt('se3aa')(torch.autograd.Variable(params))
#     jacf_1[:,k] = (tfmf_1 - tfmb_1).data / eps
#     params_v_1[k] -= eps # Reset
# diff_1 = jacf_1 - jacb_1
# print('Diff [SE3TORT]: ', diff_1.max(), diff_1.min())
