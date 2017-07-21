import torch
import torch.nn as nn
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
    if se3_dim == 'se3quat':
        se3_dim = 7
    elif se3_dim == 'affine':
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

### Pose-Mask Encoder
# Model that takes in "depth/point cloud" to generate "k"-channel masks and "k" poses represented as [R|t]
class PoseMaskEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, use_kinchain=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu'):
        super(PoseMaskEncoder, self).__init__()

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        self.conv1 = BasicConv2D(input_channels, 8, kernel_size=9, stride=1, padding=4,
                                 use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = BasicConv2D(8, 16, kernel_size=7, stride=1, padding=3,
                                 use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = BasicConv2D(16, 32, kernel_size=5, stride=1, padding=2,
                                 use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = BasicConv2D(32, 64, kernel_size=3, stride=1, padding=1,
                                 use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = BasicConv2D(64, 128, kernel_size=3, stride=1, padding=1,
                                 use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.conv1x1 = BasicConv2D(128, 128, kernel_size=1, stride=1, padding=0,
                                   use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity) # 1x1, 7x10 -> 7x10
        self.deconv1 = BasicDeconv2D(128, 64, kernel_size=(3,4), stride=2, padding=(0,1),
                                     skip_add=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = BasicDeconv2D( 64, 32, kernel_size=4, stride=2, padding=1,
                                     skip_add=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = BasicDeconv2D( 32, 16, kernel_size=6, stride=2, padding=2,
                                     skip_add=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        self.deconv4 = BasicDeconv2D( 16,  8, kernel_size=6, stride=2, padding=2,
                                     skip_add=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        self.deconv5 = nn.ConvTranspose2d(8, num_se3, kernel_size=8, stride=2, padding=3)      # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (soft-mask model)
        self.maskdecoder = nn.Softmax2d()

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        self.se3decoder  = nn.Sequential(
                                nn.Linear(128*7*10, 128),
                                get_nonlinearity(nonlinearity),
                                nn.Linear(128, self.num_se3 * self.se3_dim) # Predict the SE3s from the conv-output
                           )

        # Create pose decoder (convert to r/t)
        self.posedecoder = nn.Sequential()
        self.posedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot)) # Convert to Rt
        if use_pivot:
            self.posedecoder.add_module('pivotrt', se3nn.CollapseRtPivots()) # Collapse pivots
        if use_kinchain:
            self.posedecoder.add_module('kinchain', se3nn.ComposeRt(rightToLeft=False)) # Kinematic chain

    def forward(self, x):
        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Run mask-decoder to predict masks
        m = self.conv1x1(c5)
        m = self.deconv1([m, c4])
        m = self.deconv2([m, c3])
        m = self.deconv3([m, c2])
        m = self.deconv4([m, c1])
        m = self.maskdecoder(self.deconv5(m)) # Predict final mask

        # Run pose-decoder to predict poses
        p = c5.view(-1, 128*7*10)
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Return poses and masks
        return [p, m]

# Basic Conv + Pool + BN + Non-linearity structure
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv   = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # Convolution -> Pool -> BN -> Non-linearity
    # TODO: Test variant from the Resnet-V2 paper (https://arxiv.org/pdf/1603.05027.pdf)
    # BN -> Non-linearity -> Convolution -> Pool
    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)


# Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class BasicDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_add = True, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)
        self.skip_add = skip_add

    # Deconvolution -> (Optional Skip-Add) -> BN -> Non-linearity
    # TODO: Test variant from the Resnet-V2 paper (https://arxiv.org/pdf/1603.05027.pdf)
    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, z):
        if self.skip_add:
            x = self.deconv(z[0]) + z[1] # Skip-Add the extra input
        else:
            x = self.deconv(z)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)

### Transition model
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, use_pivot=False, se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu'):
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