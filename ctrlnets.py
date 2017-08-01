import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import se3layers as se3nn

########## HELPER FUNCS
# Choose non-linearities
def get_nonlinearity(nonlinearity):
    if nonlinearity == 'prelu':
        return nn.PReLU()
    elif nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    else:
        assert False, "Unknown non-linearity: {}".format(nonlinearity)

# Get SE3 dimension based on se3 type & pivot
def get_se3_dimension(se3_type, use_pivot):
    # Get dimension (6 for se3aa, se3euler, se3spquat)
    se3_dim = 6
    if se3_type == 'se3quat':
        se3_dim = 7
    elif se3_type == 'affine':
        se3_dim = 12
    # Add pivot dimensions
    if use_pivot:
        se3_dim += 3
    return se3_dim

# MSE Loss that gives gradients w.r.t both input & target
# NOTE: This scales the loss by 0.5 while the default nn.MSELoss does not
def BiMSELoss(input, target, size_average=True):
    diff = input - target
    loss = 0.5 * diff.dot(diff)
    if size_average:
        return loss / input.nelement()
    else:
        return loss

########## MODELS

### Normalize function
def normalize(input, p=2, dim=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.
    Does:
    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm. With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation
        dim (int): the dimension to reduce
        eps (float): small value to avoid division by zero
    """
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

### Apply weight-sharpening to the masks across the channels of the input
### output = Normalize( (sigmoid(input) + noise)^p + eps )
### where the noise is sampled from a 0-mean, sig-std dev distribution (sig is increased over time),
### the power "p" is also increased over time and the "Normalize" operation does a 1-norm normalization
def sharpen_masks(input, add_noise=True, noise_std=0, pow=1):
    input = F.sigmoid(input)
    if (add_noise and noise_std > 0):
        noise = Variable(input.data.new(input.size()).normal_(mean=0.0, std=noise_std)) # Sample gaussian noise
        input = input + noise
    input = (torch.clamp(input, min=0, max=100000) ** pow) + 1e-12  # Clamp to non-negative values, raise to a power and add a constant
    return normalize(input, p=1, dim=1, eps=1e-12)  # Normalize across channels to sum to 1

### Hook for backprops of variables, just prints min, max, mean of grads along with num of "NaNs"
def variable_hook(grad, txt):
    print txt, grad.max().data[0], grad.min().data[0], grad.mean().data[0], grad.ne(grad).sum().data[0]

### Initialize the SE3 prediction layer to identity
def init_se3layer_identity(layer, num_se3=8, se3_type='se3aa'):
    layer.weight.data.uniform_(-0.001, 0.001)  # Initialize weights to near identity
    layer.bias.data.uniform_(-0.01, 0.01)  # Initialize biases to near identity
    # Special initialization for specific SE3 types
    if se3_type == 'affine':
        bs = layer.bias.data.view(num_se3, -1)
        bs.narrow(1, 3, 9).copy_(torch.eye(3).view(1, 3, 3).expand(num_se3, 3, 3))
    elif se3_type == 'se3quat':
        bs = layer.bias.data.view(num_se3, -1)
        bs.narrow(1, 6, 1).fill_(1)  # ~ [0,0,0,1]

### Pose-Mask Encoder
# Model that takes in "depth/point cloud" to generate "k"-channel masks and "k" poses represented as [R|t]
class PoseMaskEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, use_kinchain=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1):
        super(PoseMaskEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = PreConv2D if pre_conv else BasicConv2D
        DeconvType = PreDeconv2D if pre_conv else BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        self.conv1 = ConvType(input_channels, 8, kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = ConvType(8, 16, kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = ConvType(16, 32, kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = ConvType(32, 64, kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = ConvType(64, 128, kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.conv1x1 = ConvType(128, 128, kernel_size=1, stride=1, padding=0,
                                use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity) # 1x1, 7x10 -> 7x10
        self.deconv1 = DeconvType(128, 64, kernel_size=(3,4), stride=2, padding=(0,1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = DeconvType( 64, 32, kernel_size=4, stride=2, padding=1,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = DeconvType( 32, 16, kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        self.deconv4 = DeconvType( 16,  8, kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        if pre_conv:
            # Can fit an extra BN + Non-linearity
            self.deconv5 = DeconvType(8, num_se3, kernel_size=8, stride=2, padding=3,
                                      use_bn=use_bn, nonlinearity=nonlinearity) # 8x8, 120x160 -> 240x320
        else:
            self.deconv5 = nn.ConvTranspose2d(8, num_se3, kernel_size=8, stride=2, padding=3)  # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (wt-sharpening vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate # Rate for sharpening
            self.maskdecoder = sharpen_masks # Use the weight-sharpener
        else:
            self.maskdecoder = nn.Softmax2d() # SoftMax normalization

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        self.se3decoder  = nn.Sequential(
                                nn.Linear(128*7*10, 128),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(128, self.num_se3 * self.se3_dim) # Predict the SE3s from the conv-output
                           )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the pose-mask model to predict identity transform")
            layer = self.se3decoder[2]  # Get final SE3 prediction module
            init_se3layer_identity(layer, num_se3, se3_type)  # Init to identity

        # Create pose decoder (convert to r/t)
        self.posedecoder = nn.Sequential()
        self.posedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot)) # Convert to Rt
        if use_pivot:
            self.posedecoder.add_module('pivotrt', se3nn.CollapseRtPivots()) # Collapse pivots
        if use_kinchain:
            self.posedecoder.add_module('kinchain', se3nn.ComposeRt(rightToLeft=False)) # Kinematic chain

    def compute_wt_sharpening_stats(self, train_iter=0):
        citer = 1 + (train_iter - self.sharpen_start_iter)
        noise_std, pow = 0, 1
        if (citer > 0):
            noise_std = min((citer/125000.0) * self.sharpen_rate, 0.1) # Should be 0.1 by ~12500 iters from start (if rate=1)
            pow = min(1 + (citer/500.0) * self.sharpen_rate, 100) # Should be 26 by ~12500 iters from start (if rate=1)
        return noise_std, pow

    def forward(self, x, train_iter=0):
        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Run mask-decoder to predict a smooth mask
        m = self.conv1x1(c5)
        m = self.deconv1(m, c4)
        m = self.deconv2(m, c3)
        m = self.deconv3(m, c2)
        m = self.deconv4(m, c1)
        m = self.deconv5(m)

        # Predict a mask (either wt-sharpening or soft-mask approach)
        # Normalize to sum across 1 along the channels
        if self.use_wt_sharpening:
            noise_std, pow = self.compute_wt_sharpening_stats(train_iter=train_iter)
            m = self.maskdecoder(m, add_noise=self.training, noise_std=noise_std, pow=pow)
        else:
            m = self.maskdecoder(m)

        # Run pose-decoder to predict poses
        p = c5.view(-1, 128*7*10)
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Return poses and masks
        return [p, m]

### Basic Conv + Pool + BN + Non-linearity structure
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv   = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # Convolution -> Pool -> BN -> Non-linearity
    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

### Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class BasicDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if y is not None:
            x = self.deconv(x) + y # Skip-Add the extra input
        else:
            x = self.deconv(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

###  BN + Non-linearity + Basic Conv + Pool structure
class PreConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu', **kwargs):
        super(PreConv2D, self).__init__()
        self.bn     = nn.BatchNorm2d(in_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)
        self.conv   = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None

    # BN -> Non-linearity -> Convolution -> Pool
    def forward(self, x):
        if self.bn:
            x = self.bn(x)
        x = self.nonlin(x)
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        return x

### Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class PreDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, nonlinearity='prelu', **kwargs):
        super(PreDeconv2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if self.bn:
            x = self.bn(x)
        x = self.nonlin(x)
        if y is not None:
            x = self.deconv(x) + y  # Skip-Add the extra input
        else:
            x = self.deconv(x)
        return x

### Transition model
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, use_pivot=False, se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False):
        super(TransitionModel, self).__init__()
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3

        # Pose encoder
        self.poseencoder = nn.Sequential(
                                nn.Linear(self.num_se3 * 12, 128),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(128, 256),
                                get_nonlinearity(nonlinearity)
                            )

        # Control encoder
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, 64),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(64, 128),
                                get_nonlinearity(nonlinearity)
                            )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(256+128, 256),
            get_nonlinearity(nonlinearity),
            nn.Linear(256, 128),
            get_nonlinearity(nonlinearity),
            nn.Linear(128, self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            layer = self.deltase3decoder[4]  # Get final SE3 prediction module
            init_se3layer_identity(layer, num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.deltaposedecoder = nn.Sequential()
        self.deltaposedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot))  # Convert to Rt
        if use_pivot:
            self.deltaposedecoder.add_module('pivotrt', se3nn.CollapseRtPivots())  # Collapse pivots
        if use_kinchain:
            self.deltaposedecoder.add_module('kinchain', se3nn.ComposeRt(rightToLeft=False))  # Kinematic chain

        # Compose deltas with prev pose to get next pose
        # It predicts the delta transform of the object between time "t1" and "t2": p_t2_to_t1: takes a point in t2 and converts it to t1's frame of reference
	    # So, the full transform is: p_t2 = p_t1 * p_t2_to_t1 (or: p_t2 (t2 to cam) = p_t1 (t1 to cam) * p_t2_to_t1 (t2 to t1))
        self.posedecoder = se3nn.ComposeRtPair()

    def forward(self, x):
        # Run the forward pass
        p, c = x # Pose, Control
        pv = p.view(-1, self.num_se3*12) # Reshape pose
        pe = self.poseencoder(pv)    # Encode pose
        ce = self.ctrlencoder(c)     # Encode ctrl
        x = torch.cat([pe,ce], 1)    # Concatenate encoded vectors
        x = self.deltase3decoder(x)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)
        x = self.deltaposedecoder(x) # Convert delta-SE3 to delta-Pose
        y = self.posedecoder(x, p) # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)

        # Return
        return [x, y] # Return both the deltas and the composed next pose

### SE3-Pose-Model
class SE3PoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1):
        super(SE3PoseModel, self).__init__()

        # Initialize the pose-mask model
        self.posemaskmodel = PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                             use_kinchain=use_kinchain, input_channels=input_channels,
                                             init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                             nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                             sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate)
        # Initialize the transition model
        self.transitionmodel = TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden)

    # Forward pass through the model
    def forward(self, x, train_iter=0):
        # Get input vars
        ptcloud_1, ptcloud_2, ctrl_1 = x

        # Get pose & mask predictions @ t0 & t1
        pose_1, mask_1 = self.posemaskmodel(ptcloud_1, train_iter=train_iter)  # ptcloud @ t1
        pose_2, mask_2 = self.posemaskmodel(ptcloud_2, train_iter=train_iter)  # ptcloud @ t2

        # Get transition model predicton of pose_1
        deltapose_t_12, pose_t_2 = self.transitionmodel([pose_1, ctrl_1])  # Predicts [delta-pose, pose]

        # Return outputs
        return [pose_1, mask_1], [pose_2, mask_2],  [deltapose_t_12, pose_t_2]
