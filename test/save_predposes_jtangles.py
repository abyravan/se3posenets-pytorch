# Global imports
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import time

# Torch imports
import torch
import torch.optim
import torch.utils.data
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import data
import ctrlnets
import se3posenets
import se2nets
import util
from util import AverageMeter, Tee, DataEnumerator

#### Setup options
# Common
import configargparse
parser = configargparse.ArgumentParser(description='Save network predicted poses')
parser.add_argument('--resume', default='', type=str, metavar='PATH', required=True,
                    help='path to saved checkpoint with data (default: '')')
parser.add_argument('--save-filename', default='', type=str, required=True,
                    metavar='PATH', help='file name to save results in (default: '')')
parser.add_argument('--encoder-only', action='store_true', default=False,
                    help='model we are loading is just the pose-mask encoder, '
                         'not the full SE3-Pose-Net (default: False)')
parser.add_argument('--use-data-loader', action='store_true', default=False,
                    help='Use the multi-threaded data loader (default: False)')

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ Data loading code
### Load baxter sequence from disk
def read_baxter_sequence_from_disk(dataset, id, img_ht=240, img_wd=320, img_scale=1e-4,
                                   ctrl_type='actdiffvel', num_ctrl=7,
                                   mesh_ids=torch.Tensor(),
                                   compute_bwdflows=True, load_color=None, num_tracker=0,
                                   dathreshold=0.01, dawinsize=5, use_only_da=False,
                                   noise_func=None, compute_normals=False, maxdepthdiff=0.05,
                                   bismooth_depths=False, bismooth_width=9, bismooth_std=0.001,
                                   compute_bwdnormals=False, supervised_seg_loss=False):
    # Setup vars
    num_meshes = mesh_ids.nelement()  # Num meshes
    seq_len, step_len = dataset['seq'], dataset['step']  # Get sequence & step length
    camera_intrinsics, camera_extrinsics, ctrl_ids = dataset['camintrinsics'], dataset['camextrinsics'], dataset['ctrlids']

    # Setup memory
    sequence, path, folid = data.generate_baxter_sequence(dataset, id)  # Get the file paths
    points         = torch.FloatTensor(seq_len, 3, img_ht, img_wd)
    actctrlconfigs = torch.FloatTensor(seq_len, num_ctrl)  # Ids in actual data belonging to commanded data
    poses          = torch.FloatTensor(seq_len, mesh_ids.nelement() + 1, 3, 4).zero_()
    allposes       = torch.FloatTensor()

    # Setup temp var for depth
    depths = points.narrow(1, 2, 1)  # Last channel in points is the depth

    # Setup vars for color image
    if load_color:
        rgbs = torch.ByteTensor(seq_len, 3, img_ht, img_wd)

    ## Read camera extrinsics (can be separate per dataset now!)
    try:
        camera_extrinsics = data.read_cameradata_file(path + '/cameradata.txt')
    except:
        pass  # Can use default cam extrinsics for the entire dataset

    #####
    # Load sequence
    for k in xrange(len(sequence)-1): ## Only do this for the first element!
        # Get data table
        s = sequence[k]

        # Load depth
        depths[k] = data.read_depth_image(s['depth'], img_ht, img_wd, img_scale)  # Third channel is depth (x,y,z)

        # Load configs
        state = data.read_baxter_state_file(s['state1'])
        actctrlconfigs[k] = state['actjtpos'][ctrl_ids]  # Get states for control IDs

        # Load RGB
        if load_color:
            rgbs[k] = data.read_color_image(s['color'], img_ht, img_wd, colormap=load_color)
            # actctrlvels[k] = state['actjtvel'][ctrl_ids] # Get vels for control IDs
            # comvels[k] = state['comjtvel']

        # Load SE3 state & get all poses
        se3state = data.read_baxter_se3state_file(s['se3state1'])
        if allposes.nelement() == 0:
            allposes.resize_(seq_len, len(se3state) + 1, 3, 4).fill_(0)  # Setup size
        allposes[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        for id, tfm in se3state.items():
            se3tfm = torch.mm(camera_extrinsics['modelView'],
                              tfm)  # NOTE: Do matrix multiply, not * (cmul) here. Camera data is part of options
            allposes[k][id] = se3tfm[0:3, :]  # 3 x 4 transform (id is 1-indexed already, 0 is BG)

        # Get poses of meshes we are moving
        poses[k, 0, :, 0:3] = torch.eye(3).float()  # Identity transform for BG
        for j in xrange(num_meshes):
            meshid = mesh_ids[j]
            poses[k][j + 1] = allposes[k][meshid][0:3, :]  # 3 x 4 transform

    # Compute x & y values for the 3D points (= xygrid * depths)
    xy = points[:, 0:2]
    xy.copy_(camera_intrinsics['xygrid'].expand_as(xy))  # = xygrid
    xy.mul_(depths.expand(seq_len, 2, img_ht, img_wd))  # = xygrid * depths

    # Return loaded data
    dataout = {'points': points, 'folderid': folid,
               'poses': poses, 'actctrlconfigs': actctrlconfigs}
    if load_color:
        dataout['rgbs'] = rgbs
    return dataout


################ MAIN
#@profile
def main():
    # Parse args
    global pargs, args, num_train_iter
    pargs = parser.parse_args()

    ######
    # Load the checkpoint
    assert(os.path.isfile(pargs.resume))
    # Get the checkpoint and args
    print("=> loading checkpoint '{}'".format(pargs.resume))
    checkpoint   = torch.load(pargs.resume)
    args         = checkpoint['args']
    args.start_epoch = checkpoint['epoch']
    num_train_iter   = checkpoint['train_iter']
    print("=> loaded checkpoint '{}' (epoch {}, train iter {})".format(
        pargs.resume, checkpoint['epoch'], num_train_iter))
    best_loss    = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
    best_epoch   = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_loss, best_epoch))

    # Fix some options
    if not hasattr(args, 'trans_type'):
        args.trans_type, args.posemask_type = 'default', 'default'
    if not hasattr(args, 'normal_wt'):
        args.normal_wt = 0
        args.normal_max_depth_diff = 0.05
        args.bilateral_depth_smoothing = False
        args.bilateral_depth_std = 0.005
        args.bilateral_window_width = 9
    if not hasattr(args, 'seg_wt'):
        args.seg_wt = 0

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Image suffix
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat', 'se3aar']), 'Unknown SE3 type: ' + args.se3_type
    print('Predicting {} SE3s of type: {}'.format(args.num_se3, args.se3_type))

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    ########################
    ############ Load datasets
    # Get datasets
    load_color = None
    if args.use_xyzrgb:
        load_color = 'rgb'
    elif args.use_xyzhue:
        load_color = 'hsv'
    if args.reject_left_motion:
        print("Examples where any joint of the left arm moves by > 0.005 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.reject_right_still:
        print("Examples where no joint of the right arm move by > 0.015 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.add_noise:
        print("Adding noise to the depths, actual configs & ctrls")

    ### Box dataset (vs) Other options
    valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                                         mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                                         reject_left_motion=args.reject_left_motion,
                                                                         reject_right_still=args.reject_right_still)
    read_seq_func = read_baxter_sequence_from_disk

    ### Noise function
    noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                     defprob=0.005, noisestd=0.005)

    ### Load functions
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per,
                                                         valid_filter = valid_filter,
                                                         cam_extrinsics=args.cam_extrinsics,
                                                         cam_intrinsics=args.cam_intrinsics,
                                                         ctrl_ids=args.ctrl_ids,
                                                         state_labels=args.state_labels,
                                                         add_noise=args.add_noise_data)
    disk_read_func  = lambda d, i: read_seq_func(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                 img_scale = args.img_scale, ctrl_type = args.ctrl_type,
                                                 num_ctrl=args.num_ctrl,
                                                 mesh_ids = args.mesh_ids,
                                                 compute_bwdflows=args.use_gt_masks,
                                                 dathreshold=args.da_threshold, dawinsize=args.da_winsize,
                                                 use_only_da=args.use_only_da_for_flows,
                                                 noise_func=noise_func,
                                                 load_color=load_color,
                                                 compute_normals=(args.normal_wt > 0),
                                                 maxdepthdiff=args.normal_max_depth_diff,
                                                 bismooth_depths=args.bilateral_depth_smoothing,
                                                 bismooth_width=args.bilateral_window_width,
                                                 bismooth_std=args.bilateral_depth_std,
                                                 supervised_seg_loss=(args.seg_wt > 0)) # Need BWD flows / masks if using GT masks
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    ########################
    ############ Load models & optimization stuff

    ### Load the model
    num_train_iter = 0
    num_input_channels = 3 # Num input channels
    if args.use_xyzrgb:
        num_input_channels = 6
    elif args.use_xyzhue:
        num_input_channels = 4 # Use only hue as input

    # Model
    if pargs.encoder_only:
        print("Loading only the Pose-Mask encoder from the SE3-Pose-Net")
        model = se3posenets.PoseMaskEncoder(
                    num_se3=args.num_se3, se3_type=args.se3_type,
                    input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                    init_se3_iden=args.init_posese3_iden,
                    use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                    sharpen_rate=args.sharpen_rate, wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                    num_state=args.num_state_net, noise_stop_iter=args.noise_stop_iter,
                    use_se3nn=args.use_se3nn)
    else:
        if args.use_gt_masks:
            print('Using GT masks. Model predicts only poses & delta-poses')
            assert not args.use_gt_poses, "Cannot set option for using GT masks and poses together"
            modelfn = se2nets.MultiStepSE2OnlyPoseModel if args.se2_data else ctrlnets.MultiStepSE3OnlyPoseModel
        elif args.use_gt_poses:
            print('Using GT poses & delta poses. Model predicts only masks')
            assert not args.use_gt_masks, "Cannot set option for using GT masks and poses together"
            modelfn = se2nets.MultiStepSE2OnlyMaskModel if args.se2_data else ctrlnets.MultiStepSE3OnlyMaskModel
        else:
            print("Loading the SE3-Pose-Net")
            modelfn = se2nets.MultiStepSE2PoseModel if args.se2_data else ctrlnets.MultiStepSE3PoseModel
        model = modelfn(num_ctrl=args.num_ctrl, num_se3=args.num_se3,
                        se3_type=args.se3_type, delta_pivot=args.delta_pivot, use_kinchain=False,
                        input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                        init_posese3_iden=args.init_posese3_iden, init_transse3_iden=args.init_transse3_iden,
                        use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                        sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv, decomp_model=args.decomp_model,
                        use_sigmoid_mask=args.use_sigmoid_mask, local_delta_se3=args.local_delta_se3,
                        wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                        use_jt_angles_trans=args.use_jt_angles_trans, num_state=args.num_state_net,
                        full_res=args.full_res, noise_stop_iter=args.noise_stop_iter,
                        trans_type=args.trans_type, posemask_type=args.posemask_type) # noise_stop_iter not available for SE2 models
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    # Load pre-trained weights
    model.load_state_dict(checkpoint['model_state_dict'])

    ######
    ## Iterate over train/val/test set and save the data
    datakeys = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}
    posedata = {}
    for key, dataset in datakeys.items():
        if pargs.use_data_loader:
            sampler = torch.utils.data.dataloader.SequentialSampler(dataset)  # Run sequentially along the test dataset
            loader = DataEnumerator(util.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, sampler=sampler,
                                                    pin_memory=args.use_pin_memory,
                                                    collate_fn=dataset.collate_batch))
            posedata[key] = iterate(loader, model, key)
            del loader # Delete the data loader
        else:
            posedata[key] = iterate(dataset, model, key)

    # Save data
    savedata = {'pargs': pargs, 'posedata': posedata} # Save planning args as well
    torch.save(savedata, pargs.save_filename)

### Main iterate function (train/test/val)
def iterate(dataset, model, mode='test'):
    # Get global stuff?
    global pargs, num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time  = AverageMeter(), AverageMeter()

    # Switch model modes
    model.eval()

    # Read data
    gt_poses_d, pred_poses_d, jtangles_d = [], [], []
    dfids = {}

    # Iterate over all the examples
    print('========== Dataset: {}, Num iters: {} =========='.format(mode, len(dataset)))
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    total_start = time.time()
    for i in xrange(len(dataset)):
        # ============ Load data ============ #
        # Start timer
        start = time.time()

        # Get a sample
        if pargs.use_data_loader:
            _, sample = dataset.next()
        else:
            sample = dataset[i] # Get sample
            if not sample['poses'].eq(sample['poses']).all():
                print("Sample has NaN in ground truth poses")
                continue

        # Get inputs and targets (as variables)
        # B x S x C x H x W if using data loader or S x C x H x W
        pts      = util.to_var(sample['points'].type(deftype), volatile=True)

        # Get XYZRGB input
        chdim = 2 if pargs.use_data_loader else 1 # It is B x S x C x H x W if we use data loader
        if args.use_xyzrgb:
            rgb = util.to_var(sample['rgbs'].type(deftype)/255.0, requires_grad=False) # Normalize RGB to 0-1
            netinput = torch.cat([pts, rgb], chdim) # Concat along channels dimension
        elif args.use_xyzhue:
            hue = util.to_var(sample['rgbs'].narrow(chdim,0,1).type(deftype)/179.0, requires_grad=False)  # Normalize Hue to 0-1 (Opencv has hue from 0-179)
            netinput = torch.cat([pts, hue], chdim) # Concat along channels dimension
        else:
            netinput = pts # XYZ

        # Get jt angles
        jtangles = util.to_var(sample['actctrlconfigs'].type(deftype), requires_grad=False) #[:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============ #
        # Start timer
        start = time.time()

        ### Predict the pose from the network
        poseprednet = model if pargs.encoder_only else model.posemaskmodel
        inp = [netinput[:,0], jtangles[:,0]] if args.use_jt_angles else netinput[:,0]
        if pargs.use_data_loader:
            pred_poses = poseprednet(inp, predict_masks=False)
        else:
            pred_poses = poseprednet(inp, predict_masks=False)

        # Measure fwd pass time
        fwd_time.update(time.time() - start)

        # ============ Save stuff ============ #
        bsz = pts.size(0) if pargs.use_data_loader else 1
        for kk in xrange(bsz):
            did = sample['datasetid'][kk] if pargs.use_data_loader else sample['datasetid']  # Dataset ID
            fid = sample['folderid'][kk] if pargs.use_data_loader else sample['folderid']    # Folder ID in dataset
            if (did, fid) not in dfids:
                print('Added new ID ({}: {}, {})'.format(len(dfids), did, fid))
                dfids[(did, fid)] = len(dfids)  # We have seen this pair
                pred_poses_d.append([])
                gt_poses_d.append([])
                jtangles_d.append([])
            # Save to list
            if pargs.use_data_loader:
                pred_poses_d[-1].append(pred_poses.data[kk:kk+1].cpu().clone())
                gt_poses_d[-1].append(sample['poses'][kk:kk+1,0].clone()) # Choose one element only, 1 x 8 x 3 x 4
                jtangles_d[-1].append(sample['actctrlconfigs'][kk:kk+1,0].clone()) # 1 x 7
            else:
                pred_poses_d[-1].append(pred_poses.data.cpu().clone())
                gt_poses_d[-1].append(sample['poses'][0:1].clone())  # Choose one element only, 1 x 8 x 3 x 4
                jtangles_d[-1].append(sample['actctrlconfigs'][0:1].clone())  # 1 x 7

        # Print stats
        if i % (1000 if pargs.use_data_loader else 5000) == 0:
            print('Dataset: {}, Data-Folder ID: {}, Example: {}/{}'.format(mode, len(pred_poses_d), i+1, len(dataset)))
            print('\tTime => Total: {:.3f}, Data: {data.val:.3f} ({data.avg:.3f}), '
                  'Fwd: {fwd.val:.3f} ({fwd.avg:.3f})'.format(
                time.time() - total_start, data=data_time, fwd=fwd_time))

    ## Concat stuff and return
    for kk in xrange(len(gt_poses_d)):
        gt_poses_d[kk]   = torch.cat(gt_poses_d[kk], 0)    # N x 8 x 3 x 4
        pred_poses_d[kk] = torch.cat(pred_poses_d[kk], 0)  # N x 8 x 3 x 4
        jtangles_d[kk] = torch.cat(jtangles_d[kk], 0)      # N x 7
    return {'gtposes': gt_poses_d, 'predposes': pred_poses_d, 'jtangles': jtangles_d}


################ RUN MAIN
if __name__ == '__main__':
    main()
