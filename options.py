import configargparse

def setup_comon_options():
    # Parse arguments
    parser = configargparse.ArgumentParser(description='SE3-Pose-Nets Training')

    # Dataset options
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='Path to config file for parameters')
    parser.add_argument('-d', '--data', default=[], required=True,
                        action='append', metavar='DIRS', help='path to dataset(s), passed in as list [a,b,c...]')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-per', default=0.6, type=float,
                        metavar='FRAC', help='fraction of data for the training set (default: 0.6)')
    parser.add_argument('--val-per', default=0.15, type=float,
                        metavar='FRAC', help='fraction of data for the validation set (default: 0.15)')
    parser.add_argument('--img-scale', default=1e-4, type=float,
                        metavar='IS', help='conversion scalar from depth resolution to meters (default: 1e-4)')
    parser.add_argument('--img-suffix', default='sub', type=str,
                        metavar='SUF', help='image suffix for getting file names of images on disk (default: "sub")')
    parser.add_argument('--step-len', default=1, type=int,
                        metavar='N', help='number of frames separating each example in the training sequence (default: 1)')
    parser.add_argument('--seq-len', default=1, type=int,
                        metavar='N', help='length of the training sequence (default: 1)')
    parser.add_argument('--ctrl-type', default='actdiffvel', type=str,
                        metavar='STR', help='Control type: actvel | actacc | comvel | comacc | [actdiffvel] | comdiffvel. '
                                            'For box dataset: ballposforce | ballposeforce | ballposvelforce | ballposevelforce')
    parser.add_argument('--num-ctrl', default=7, type=int, metavar='N',
                        help='dimensionality of the control space (default: 7)')
    parser.add_argument('--se2-data', action='store_true', default=False,
                        help='SE2 data. (default: False)')
    parser.add_argument('--mean-dt', default=0.0, type=float, metavar='DT',
                        help='Mean expected time-difference between two consecutive frames in the dataset.'
                             'If not set, will revert to step_len * (1.0/30.0) ~ 30 fps camera (default: 0)')
    parser.add_argument('--std-dt', default=0.005, type=float, metavar='DT',
                        help='Std.deviaton of the time-difference between two consecutive frames in the dataset.'
                             'All examples that have dt over 2*std_dts from the mean will be discarded (default: 0.005 seconds)')
    parser.add_argument('--da-threshold', default=0.01, type=float, metavar='DIST',
                        help='Threshold for DA (used for flow/visibility computation) (default: 0.01 m)')
    parser.add_argument('--da-winsize', default=5, type=int, metavar='WIN',
                        help='Windowsize for DA search (used for flow/visibility computation) (default: 5)')
    parser.add_argument('--use-only-da-for-flows', action='store_true', default=False,
                        help='Use only data-association for computing flows, dont use tracker info. (default: False)')
    parser.add_argument('--reject-left-motion', action='store_true', default=False,
                        help='Reject examples where any joint of the left arm moves by >0.005 radians inter-frame. (default: False)')
    parser.add_argument('--reject-right-still', action='store_true', default=False,
                        help='Reject examples where all joints of the right arm move by <0.01 radians inter-frame. (default: False)')
    parser.add_argument('--add-noise', action='store_true', default=False,
                        help='Enable adding noise to the depths and the configs/ctrls. (default: False)')
    parser.add_argument('--add-noise-data', default=[], required=False,
                        action='append', metavar='DIRS', help='noise setting per dataset. has to correspond to number in --data [a,b,c...]')

    # New options
    parser.add_argument('--full-res', action='store_true', default=False,
                        help='Full-resolution input images -> 480x640 (default: False)')
    parser.add_argument('--use-xyzrgb', action='store_true', default=False,
                        help='Use RGB as input along with XYZ -> XYZRGB input (default: False)')
    parser.add_argument('--use-xyzhue', action='store_true', default=False,
                        help='Use hue channel (from HSV) as input along with XYZ -> XYZHUE input (default: False)')

    # Model options
    parser.add_argument('--no-batch-norm', action='store_true', default=False,
                        help='disables batch normalization (default: False)')
    parser.add_argument('--pre-conv', action='store_true', default=False,
                        help='puts batch normalization and non-linearity before the convolution / de-convolution (default: False)')
    parser.add_argument('--nonlin', default='prelu', type=str,
                        metavar='NONLIN', help='type of non-linearity to use: [prelu] | relu | tanh | sigmoid | elu | selu')
    parser.add_argument('--se3-type', default='se3aa', type=str,
                        metavar='SE3', help='SE3 parameterization: [se3aa] | se3quat | se3spquat | se3euler | affine | se3aar')
    parser.add_argument('-n', '--num-se3', type=int, default=8,
                        help='Number of SE3s to predict (default: 8)')
    parser.add_argument('--init-transse3-iden', action='store_true', default=False,
                        help='Initialize the weights for the SE3 prediction layer of the transition model to predict identity')
    parser.add_argument('--init-posese3-iden', action='store_true', default=False,
                        help='Initialize the weights for the SE3 prediction layer of the pose-mask model to predict identity')
    parser.add_argument('--local-delta-se3', action='store_true', default=False,
                        help='Predicted delta-SE3 operates in local co-ordinates not global co-ordinates, '
                             'so if we predict "D", full-delta = P1 * D * P1^-1, P2 = P1 * D')
    parser.add_argument('--use-ntfm-delta', action='store_true', default=False,
                        help='Uses the variant of the NTFM3D layer that computes the weighted avg. delta')
    parser.add_argument('--wide-model', action='store_true', default=False,
                        help='Wider network')
    parser.add_argument('--decomp-model', action='store_true', default=False,
                        help='Use a separate encoder for predicting the pose and masks')
    parser.add_argument('--use-gt-masks', action='store_true', default=False,
                        help='Model predicts only poses & delta poses. GT masks are given. (default: False)')
    parser.add_argument('--use-gt-poses', action='store_true', default=False,
                        help='Model predicts only masks. GT poses & deltas are given. (default: False)')
    parser.add_argument('--use-jt-angles', action='store_true', default=False,
                        help='Model uses GT jt angles as inputs to the pose net. (default: False)')
    parser.add_argument('--use-jt-angles-trans', action='store_true', default=False,
                        help='Model uses GT jt angles as inputs to the transition net. (default: False)')

    # Mask options
    parser.add_argument('--use-wt-sharpening', action='store_true', default=False,
                        help='use weight sharpening for the mask prediction (instead of the soft-mask model) (default: False)')
    parser.add_argument('--sharpen-start-iter', default=0, type=int,
                        metavar='N', help='Start the weight sharpening from this training iteration (default: 0)')
    parser.add_argument('--sharpen-rate', default=1.0, type=float,
                        metavar='W', help='Slope of the weight sharpening (default: 1.0)')
    parser.add_argument('--noise-stop-iter', default=1e6, type=int,
                        metavar='N', help='Stop noise addition during weight sharpening from this training iteration(default: 1e6)')
    parser.add_argument('--use-sigmoid-mask', action='store_true', default=False,
                        help='treat each mask channel independently using the sigmoid non-linearity. Pixel can belong to multiple masks (default: False)')
    parser.add_argument('--soft-wt-sharpening', action='store_true', default=False,
                        help='Uses soft loss + weight sharpening (default: False)')

    # Loss options
    parser.add_argument('--loss-type', default='mse', type=str,
                        metavar='STR', help='Type of loss to use for 3D point errors, only works if we are not using '
                                            'soft-masks (default: mse | abs, normmsesqrt, normmsesqrtpt )')
    parser.add_argument('--motion-norm-loss', action='store_true', default=False,
                        help='normalize the losses by number of points that actually move instead of size average (default: False)')
    parser.add_argument('--consis-wt', default=0.1, type=float,
                        metavar='WT', help='Weight for the pose consistency loss (default: 0.1)')
    parser.add_argument('--loss-scale', default=10000, type=float,
                        metavar='WT', help='Default scale factor for all the losses (default: 1000)')
    parser.add_argument('--pose-dissim-wt', default=0.0, type=float,
                        metavar='WT', help='Weight for the dissimilarity loss in the pose space (default: 0)')
    parser.add_argument('--delta-dissim-wt', default=0.0, type=float,
                        metavar='WT', help='Weight for the loss that regularizes the predicted delta away from zero (default: 0)')
    parser.add_argument('--no-consis-delta-grads', action='store_true', default=False,
                        help="Don't backpropagate the consistency gradients to the predicted deltas. (default: False)")

    # Training options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--use-pin-memory', action='store_true', default=False,
                        help='Use pin memory - note that this uses additional CPU RAM (default: False)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--train-ipe', default=2000, type=int, metavar='N',
                        help='number of training iterations per epoch (default: 2000)')
    parser.add_argument('--val-ipe', default=500, type=int, metavar='N',
                        help='number of validation iterations per epoch (default: 500)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Optimization options
    parser.add_argument('-o', '--optimization', default='adam', type=str,
                        metavar='OPTIM', help='type of optimization: sgd | [adam]')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-decay', default=0.1, type=float, metavar='M',
                        help='Decay learning rate by this value every decay-epochs (default: 0.1)')
    parser.add_argument('--decay-epochs', default=30, type=int,
                        metavar='M', help='Decay learning rate every this many epochs (default: 10)')
    parser.add_argument('--min-lr', default=1e-5, type=float,
                        metavar='LR', help='min learning rate (default: 1e-5)')

    # Display/Save options
    parser.add_argument('--disp-freq', '-p', default=25, type=int,
                        metavar='N', help='print/disp/save frequency (default: 25)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-s', '--save-dir', default='results', type=str,
                        metavar='PATH', help='directory to save results in. If it doesnt exist, will be created. (default: results/)')
    parser.add_argument('--disp-err-per-mask', action='store_true', default=False,
                        help='Display flow error per mask channel. (default: False)')
    parser.add_argument('--reset-train-iter', action='store_true', default=False,
                        help='Reset num_train_iter to 0 -> for weight sharpening (default: False)')
    parser.add_argument('--detailed-test-stats', action='store_true', default=False,
                        help='Save detailed test statistics if this option is set (default: False)')

    ###############
    ## Deprecated
    parser.add_argument('--pred-pivot', action='store_true', default=False,
                        help='Predict pivot in addition to the SE3 parameters (default: False)')

    # Return
    return parser