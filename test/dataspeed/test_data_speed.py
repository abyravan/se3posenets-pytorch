# Add path to necessary packages
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '../../')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Global imports
import time

# Torch imports
import torch
import torch.optim
import torch.utils.data

# Local imports
import data
import util
from util import AverageMeter, DataEnumerator

#### Setup options
# Common
import options
parser = options.setup_comon_options()
parser.add_argument('-m', '--num', default=100, type=int,
                    metavar='N', help='Number of batches to load (default: 100)')

################ MAIN
#@profile
def main():
    # Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup extra options
    args.img_ht, args.img_wd, args.img_suffix = 240, 320, 'sub'
    args.num_ctrl = 14 if (args.ctrl_type.find('both') >= 0) else 7 # Number of control dimensions
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data
    load_dir = args.data[0] #args.data.split(',,')[0]
    args.baxter_labels = data.read_baxter_labels_file(load_dir + '/statelabels.txt')
    args.mesh_ids      = args.baxter_labels['meshIds']
    args.cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

    # Camera parameters
    args.cam_intrinsics = {'fx': 589.3664541825391/2,
                           'fy': 589.3664541825391/2,
                           'cx': 320.5/2,
                           'cy': 240.5/2}
    args.cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
                                                                               args.cam_intrinsics)

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    ########################
    ############ Load datasets
    # Get datasets
    baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
                                                         step_len = args.step_len, seq_len = args.seq_len,
                                                         train_per = args.train_per, val_per = args.val_per)
    disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
                                                                       img_scale = args.img_scale, ctrl_type = 'actdiffvel',
                                                                       mesh_ids = args.mesh_ids,
                                                                       camera_extrinsics = args.cam_extrinsics,
                                                                       camera_intrinsics = args.cam_intrinsics,
                                                                       compute_bwdflows=True)
    train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train') # Train dataset
    val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')   # Val dataset
    test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('Dataset size => Train: {}, Validation: {}, Test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # Create dataloader
    train_loader = DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=args.use_pin_memory,
                                        collate_fn=train_dataset.collate_batch))
    time.sleep(10) # Wait for some loading to happen initially

    # Test
    deftype = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor' # Default tensor type
    print('Loading {} batches to test speed'.format(args.num))
    data_time = AverageMeter()
    for k in xrange(args.num):
        # Time it
        st = time.time()

        # Load data
        j, sample = train_loader.next()

        # Averate time
        data_time.update(time.time()-st)

        ## Do some stuff
        # Get inputs and targets (as variables)
        pts = util.to_var(sample['points'].type(deftype), requires_grad=True)
        ctrls = util.to_var(sample['controls'].type(deftype), requires_grad=True)
        fwdflows = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        tarpts = util.to_var(sample['fwdflows'].type(deftype), requires_grad=False)
        tarpts.data.add_(pts.data.narrow(1, 0, 1).expand_as(tarpts.data))  # Add "k"-step flows to the initial point cloud

        # Pause for a bit
        time.sleep(0.25)

        # Stats
        if (k % 10 == 0) or (k == args.num-1):
            print('Loaded {}/{} batches, Total/Avg time: {}/{}'.format(k+1, args.num, data_time.sum, data_time.avg))

################ RUN MAIN
if __name__ == '__main__':
    main()
