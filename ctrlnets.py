import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import se3layers as se3nn

################################################################################
'''
    Helper functions
'''
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

### Hook for backprops of variables, just prints min, max, mean of grads along with num of "NaNs"
def variable_hook(grad, txt):
    print txt, grad.max().data[0], grad.min().data[0], grad.mean().data[0], grad.ne(grad).sum().data[0]

### Initialize the SE3 prediction layer to identity
def init_se3layer_identity(layer, num_se3=8, se3_type='se3aa'):
    layer.weight.data.uniform_(-0.001, 0.001)  # Initialize weights to near identity
    layer.bias.data.uniform_(-0.01, 0.01)  # Initialize biases to near identity
    # Special initialization for specific SE3 types
    if se3_type == 'affine':
        bs = layer.bias.data.view(num_se3, 3, 4)
        bs.narrow(2, 0, 3).copy_(torch.eye(3).view(1, 3, 3).expand(num_se3, 3, 3))
    elif se3_type == 'se3quat':
        bs = layer.bias.data.view(num_se3, -1)
        bs.narrow(1, 6, 1).fill_(1)  # ~ [0,0,0,1]

### Initialize the last deconv layer such that the mask predicts BG for first channel & zero for all other channels
def init_sigmoidmask_bg(layer, num_se3=8):
    # Init weights to be close to zero, so that biases affect the result primarily
    layer.weight.data.uniform_(-1e-3,1e-3) # sigmoid(0) ~= 0.5

    # Init first channel bias to large +ve value
    layer.bias.data.narrow(0,0,1).uniform_(4.1,3.9) # sigmoid(4) ~= 1

    # Init other layers biases to large -ve values
    layer.bias.data.narrow(0,1,num_se3-1).uniform_(-4.1,-3.9)   # sigmoid(-4) ~= 0

################################################################################
'''
    NN Modules / Loss functions
'''

# MSE Loss that gives gradients w.r.t both input & target
# NOTE: This scales the loss by 0.5 while the default nn.MSELoss does not
def BiMSELoss(input, target, size_average=True):
    diff = (input - target).view(-1)
    loss = 0.5 * diff.dot(diff)
    if size_average:
        return loss / input.nelement()
    else:
        return loss

### Basic Conv + Pool + BN + Non-linearity structure
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
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
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if y is not None:
            x = self.deconv(x) + y  # Skip-Add the extra input
        else:
            x = self.deconv(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

###  BN + Non-linearity + Basic Conv + Pool structure
class PreConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu', **kwargs):
        super(PreConv2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None

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
    return F.normalize(input, p=1, dim=1, eps=1e-12)  # Normalize across channels to sum to 1

################################################################################
'''
    Single step / Recurrent models
'''

####################################
### Pose Encoder
# Model that takes in "depth/point cloud" to generate "k" poses represented as [R|t]
class PoseEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, use_kinchain=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False):
        super(PoseEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('[PoseEncoder] Using BN + Non-Linearity + Conv/Deconv architecture')
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

    def forward(self, x):
        # Run conv-encoder to generate embedding
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Run pose-decoder to predict poses
        p = x.view(-1, 128*7*10)
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Return poses
        return p

####################################
### Mask Encoder (single encoder that takes a depth image and predicts segmentation masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks
class MaskEncoder(nn.Module):
    def __init__(self, num_se3, pre_conv=False, input_channels=3, use_bn=True, nonlinearity='prelu',
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False):
        super(MaskEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType = PreConv2D if pre_conv else BasicConv2D
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
                                use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
        self.deconv1 = DeconvType(128, 64, kernel_size=(3, 4), stride=2, padding=(0, 1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = DeconvType(64, 32, kernel_size=4, stride=2, padding=1,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = DeconvType(32, 16, kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        self.deconv4 = DeconvType(16, 8, kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        if pre_conv:
            # Can fit an extra BN + Non-linearity
            self.deconv5 = DeconvType(8, num_se3, kernel_size=8, stride=2, padding=3,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 8x8, 120x160 -> 240x320
        else:
            self.deconv5 = nn.ConvTranspose2d(8, num_se3, kernel_size=8, stride=2,
                                              padding=3)  # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (wt-sharpening vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter  # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate  # Rate for sharpening
            self.maskdecoder = sharpen_masks  # Use the weight-sharpener
        elif use_sigmoid_mask:
            #lastdeconvlayer = self.deconv5.deconv if pre_conv else self.deconv5
            #init_sigmoidmask_bg(lastdeconvlayer, num_se3) # Initialize last deconv layer to predict BG | all zeros
            self.maskdecoder = nn.Sigmoid() # No normalization, each pixel can belong to multiple masks but values between (0-1)
        else:
            self.maskdecoder = nn.Softmax2d()  # SoftMax normalization

    def compute_wt_sharpening_stats(self, train_iter=0):
        citer = 1 + (train_iter - self.sharpen_start_iter)
        noise_std, pow = 0, 1
        if (citer > 0):
            noise_std = min((citer / 125000.0) * self.sharpen_rate,
                            0.1)  # Should be 0.1 by ~12500 iters from start (if rate=1)
            pow = min(1 + (citer / 500.0) * self.sharpen_rate,
                      100)  # Should be 26 by ~12500 iters from start (if rate=1)
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

        # Predict a mask (either wt-sharpening or sigmoid-mask or soft-mask approach)
        # Normalize to sum across 1 along the channels (only for weight sharpening or soft-mask)
        if self.use_wt_sharpening:
            noise_std, pow = self.compute_wt_sharpening_stats(train_iter=train_iter)
            m = self.maskdecoder(m, add_noise=self.training, noise_std=noise_std, pow=pow)
        else:
            m = self.maskdecoder(m)

        # Return masks
        return m

####################################
### Pose-Mask Encoder (single encoder that predicts both poses and masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks and "k" poses represented as [R|t]
# NOTE: We can set a conditional flag that makes it predict both poses/masks or just poses
class PoseMaskEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, use_kinchain=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False):
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

        # Normalize to generate mask (wt-sharpening vs sigmoid-mask vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate # Rate for sharpening
            self.maskdecoder = sharpen_masks # Use the weight-sharpener
        elif use_sigmoid_mask:
            #lastdeconvlayer = self.deconv5.deconv if pre_conv else self.deconv5
            #init_sigmoidmask_bg(lastdeconvlayer, num_se3) # Initialize last deconv layer to predict BG | all zeros
            self.maskdecoder = nn.Sigmoid() # No normalization, each pixel can belong to multiple masks but values between (0-1)
        else:
            self.maskdecoder = nn.Softmax2d() # SoftMax normalization

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        sdim = 64 #128
        self.se3decoder  = nn.Sequential(
                                nn.Linear(128 * 7 * 10, sdim),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(sdim, self.num_se3 * self.se3_dim)  # Predict the SE3s from the conv-output
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

    def forward(self, x, predict_masks=True, train_iter=0):
        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Run pose-decoder to predict poses
        p = c5.view(-1, 128 * 7 * 10)
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Run mask-decoder to predict a smooth mask
        # NOTE: Conditional based on input flag
        if predict_masks:
            m = self.conv1x1(c5)
            m = self.deconv1(m, c4)
            m = self.deconv2(m, c3)
            m = self.deconv3(m, c2)
            m = self.deconv4(m, c1)
            m = self.deconv5(m)

            # Predict a mask (either wt-sharpening or sigmoid-mask or soft-mask approach)
            # Normalize to sum across 1 along the channels (only for weight sharpening or soft-mask)
            if self.use_wt_sharpening:
                noise_std, pow = self.compute_wt_sharpening_stats(train_iter=train_iter)
                m = self.maskdecoder(m, add_noise=self.training, noise_std=noise_std, pow=pow)
            else:
                m = self.maskdecoder(m)

            # Return both
            return p, m
        else:
            return p


####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, use_pivot=False, se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False):
        super(TransitionModel, self).__init__()
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3

        # Pose encoder
        pdim = [128, 256]
        self.poseencoder = nn.Sequential(
                                nn.Linear(self.num_se3 * 12, pdim[0]),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(pdim[0], pdim[1]),
                                get_nonlinearity(nonlinearity)
                            )

        # Control encoder
        cdim = [128, 256] #[64, 128]
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, cdim[0]),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(cdim[0], cdim[1]),
                                get_nonlinearity(nonlinearity)
                            )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(pdim[1]+cdim[1], 256),
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

        # In case the predicted delta (D) is in the local frame of reference, we compute the delta in the global reference
        # system in the following way:
        # SE3_2 = SE3_1 * D_local (this takes a point from the local reference frame to the global frame)
        # D_global = SE3_1 * D_local * SE3_1^-1 (this takes a point in the global reference frame, transforms it and returns a point in the same reference frame)
        self.local_delta_se3 = local_delta_se3
        if self.local_delta_se3:
            print('Deltas predicted by transition model will affect points in local frame of reference')
            self.rtinv = se3nn.RtInverse()
            self.globaldeltadecoder = se3nn.ComposeRtPair()

    def forward(self, x):
        # Run the forward pass
        p, c = x # Pose, Control
        pv = p.view(-1, self.num_se3*12) # Reshape pose
        pe = self.poseencoder(pv)    # Encode pose
        ce = self.ctrlencoder(c)     # Encode ctrl
        x = torch.cat([pe,ce], 1)    # Concatenate encoded vectors
        x = self.deltase3decoder(x)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)
        x = self.deltaposedecoder(x)  # Convert delta-SE3 to delta-Pose (can be in local or global frame of reference)
        if self.local_delta_se3:
            # Predicted delta is in the local frame of reference, can't use it directly
            z = self.posedecoder(p, x) # SE3_2 = SE3_1 * D_local (takes a point in local frame to global frame)
            y = self.globaldeltadecoder(z, self.rtinv(p)) # D_global = SE3_2 * SE3_1^-1 = SE3_1 * D_local * SE3_1^-1 (from global to global)
        else:
            # Predicted delta is already in the global frame of reference, use it directly (from global to global)
            z = self.posedecoder(x, p) # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)
            y = x # D = SE3_2 * SE3_1^-1 (global to global)

        # Return
        return [y, z] # Return both the deltas (in global frame) and the composed next pose

####################################
### SE3-Pose-Model (single-step model that takes [depth_t, depth_t+1, ctrl-t] to predict
### [pose_t, mask_t], [pose_t+1, mask_t+1], [delta-pose, poset_t+1]
class SE3PoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False):
        super(SE3PoseModel, self).__init__()

        # Initialize the pose-mask model
        self.posemaskmodel = PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                             use_kinchain=use_kinchain, input_channels=input_channels,
                                             init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                             nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                             sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                             use_sigmoid_mask=use_sigmoid_mask)
        # Initialize the transition model
        self.transitionmodel = TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden,
                                               local_delta_se3=local_delta_se3)

    # Forward pass through the model
    def forward(self, x, train_iter=0):
        # Get input vars
        ptcloud_1, ptcloud_2, ctrl_1 = x

        # Get pose & mask predictions @ t0 & t1
        pose_1, mask_1 = self.posemaskmodel(ptcloud_1, train_iter=train_iter, predict_masks=True)  # ptcloud @ t1
        pose_2, mask_2 = self.posemaskmodel(ptcloud_2, train_iter=train_iter, predict_masks=True)  # ptcloud @ t2

        # Get transition model predicton of pose_1
        deltapose_t_12, pose_t_2 = self.transitionmodel([pose_1, ctrl_1])  # Predicts [delta-pose, pose]

        # Return outputs
        return [pose_1, mask_1], [pose_2, mask_2],  [deltapose_t_12, pose_t_2]


####################################
### Multi-step version of the SE3-Pose-Model
### Currently, this has a separate pose & mask predictor as well as a transition model
### NOTE: The forward pass is not currently implemented, this needs to be done outside
class MultiStepSE3PoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False, decomp_model=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False):
        super(MultiStepSE3PoseModel, self).__init__()

        # Initialize the pose & mask model
        self.decomp_model = decomp_model
        if self.decomp_model:
            print('Using separate networks for pose and mask prediction')
            self.posemodel = PoseEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                         use_kinchain=use_kinchain, input_channels=input_channels,
                                         init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity)
            self.maskmodel = MaskEncoder(num_se3=num_se3, input_channels=input_channels,
                                         use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                         sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                         use_sigmoid_mask=use_sigmoid_mask)
        else:
            print('Using single network for pose & mask prediction')
            self.posemaskmodel = PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                                 use_kinchain=use_kinchain, input_channels=input_channels,
                                                 init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                                 nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                                 sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                                 use_sigmoid_mask=use_sigmoid_mask)

        # Initialize the transition model
        self.transitionmodel = TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden,
                                               local_delta_se3=local_delta_se3)
    # Predict pose only
    def forward_only_pose(self, x):
        if self.decomp_model:
            return self.posemodel(x)
        else:
            return self.posemaskmodel(x, predict_masks=False) # returns only pose

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        if self.decomp_model:
            pose = self.posemodel(x)
            mask = self.maskmodel(x, train_iter=train_iter)
            return pose, mask
        else:
            return self.posemaskmodel(x, train_iter=train_iter, predict_masks=True) # Predict both

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl):
        return self.transitionmodel([pose, ctrl])

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for Multi-Step SE3-Pose-Model is not yet implemented')
        raise NotImplementedError