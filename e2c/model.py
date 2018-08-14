# Torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

# Local imports
import ctrlnets

## todo: concat rgb/depth to pass through one conv layer, pass depth/rgb through separate conv layers & concat features

##### Override the multi-variate normal distribution to add a distribution with v & r terms for the E2C transition model
class MVNormal(torch.distributions.MultivariateNormal):
    def __init__(self, *args, v=None, r=None, **kwargs):
        super(MVNormal, self).__init__(*args, **kwargs)
        self.v, self.r = v, r

    def rsample_e2c(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape=sample_shape)

@torch.distributions.kl.register_kl(MVNormal, MVNormal)
def kl_mvnormal_mvnormal(p, q):
    if (p.v is not None): # Covar of p is A*\sigma*A^T where A = I + vr^T (special expression for KL from E2C paper)
        raise NotImplementedError # todo
    else:
        return torch.distributions.kl._kl_multivariatenormal_multivariatenormal(p, q)

##### Override the independent normal distribution to add a distribution with v & r terms for the E2C transition model
class Normal(torch.distributions.Normal):
    def __init__(self, *args, v=None, r=None, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.v, self.r = v, r

    def rsample_e2c(self, sample_shape=torch.Size()):
        # Use torch.distributions.normal rsample if not E2C style distribution
        if (self.v is not None):
            # todo: compute sig_1 = A * sig * A^T where A = I + vr^T and sample from MVNormal(mean, sig_1)
            raise NotImplementedError
        else:
            return self.rsample(sample_shape=sample_shape)

@torch.distributions.kl.register_kl(Normal, Normal)
def kl_normal_normal(p, q):
    if (p.v is not None):  # Covar of p is A*\sigma*A^T where A = I + vr^T (special expression for KL from E2C paper)
        raise NotImplementedError  # todo
    else:
        return torch.distributions.kl._kl_normal_normal(p, q)

#################
### Basic Conv + Pool + BN + Non-linearity structure
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, norm_type='bn', nonlinearity='prelu',
                 coord_conv=False, **kwargs):
        super(BasicConv2D, self).__init__()
        if coord_conv:
            self.conv = ctrlnets.CoordConv(in_channels, out_channels, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channels, eps=0.001)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm((out_channels, out_channels))
        else:
            self.norm = None
        self.nonlin = ctrlnets.get_nonlinearity(nonlinearity)

    # Convolution -> Pool -> BN -> Non-linearity
    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        if self.norm:
            x = self.norm(x)
        return self.nonlin(x)

### Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class BasicDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='bn', nonlinearity='prelu',
                 coord_conv=False, **kwargs):
        super(BasicDeconv2D, self).__init__()
        if coord_conv:
            self.deconv = ctrlnets.CoordConvT(in_channels, out_channels, **kwargs)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channels, eps=0.001)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm((out_channels, out_channels))
        else:
            self.norm = None
        self.nonlin = ctrlnets.get_nonlinearity(nonlinearity)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if y is not None:
            x = self.deconv(x) + y  # Skip-Add the extra input
        else:
            x = self.deconv(x)
        if self.norm:
            x = self.norm(x)
        return self.nonlin(x)

#############################
##### Create Encoder
class Encoder(nn.Module):
    def __init__(self, img_type='rgbd', norm_type='bn', nonlin_type='prelu', wide_model=False,
                 use_state=False, num_state=7, coord_conv=False,
                 conv_encode=False, deterministic=False):
        super(Encoder, self).__init__()

        # Normalization
        assert (norm_type in ['none', 'bn', 'ln', 'in']),  "Unknown normalization type input: {}".format(norm_type)

        # Coordinate convolution
        if coord_conv:
            print('[Encoder] Using co-ordinate convolutional layers')
        ConvType = lambda x, y, **v: BasicConv2D(in_channels=x, out_channels=y,
                                                 coord_conv=coord_conv, **v)

        # Input type
        print("[Encoder] Input image type: {}, Normalization type: {}, Nonliearity: {}".format(
            img_type, norm_type, nonlin_type))
        if img_type == 'rgb' or img_type == 'xyz':
            input_channels = 3
        elif img_type == 'd':
            input_channels = 1
        elif img_type == 'rgbd':
            input_channels = 4
        elif img_type == 'rgbxyz':
            input_channels = 6
        else:
            assert False, "Unknown image type input: {}".format(img_type)

        ###### Encode XYZ-RGB images
        # Create conv-encoder (large net => 5 conv layers with pooling)
        # 9x9, 240x320 -> 120x160
        # 7x7, 120x160 -> 60x80
        # 5x5, 60x80 -> 30x40
        # 3x3, 30x40 -> 15x20
        # 3x3, 15x20 -> 7x10
        chn = [input_channels, 32, 64, 128, 256, 256] if wide_model else [input_channels, 16, 32, 64, 128, 128]  # Num channels
        kern = [9,7,5,3,3] # Kernel sizes
        self.imgencoder = nn.Sequential(
            *[ConvType(chn[k], chn[k+1], kernel_size=kern[k], stride=1, padding=kern[k]//2,
                      use_pool=True, norm_type=norm_type, nonlinearity=nonlin_type) for k in range(len(chn)-1)]
        )

        ###### Encode state information (jt angles, gripper position)
        self.use_state = use_state
        if self.use_state:
            print("[Encoder] Using state as input")
            sdim = [num_state, 32, 64, 64]
            self.stateencoder = nn.Sequential(
                nn.Linear(sdim[0], sdim[1]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sdim[1], sdim[2]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sdim[2], sdim[3]),
            )
        else:
            sdim = [0]

        ###### If we are using conv output (fully-conv basically), we can do 1x1 conv stuff
        self.conv_encode = conv_encode
        self.deterministic = deterministic
        if self.conv_encode:
            # 1x1, 7x10 -> 3x5, 256
            # 1x1, 3x5 -> 3x5, 128
            # 1x1, 3x5 -> 3x5, 64
            # 1x1, 3x5 -> 3x5, 32
            print("[Encoder] Using convolutional network with 1x1 convolutions for state prediction")
            chn_out = [sdim[-1]+chn[-1], 256, 128, 64, 32] if wide_model else [sdim[-1]+chn[-1], 128, 64, 32, 16]
            pool = [True, False, False, False] # Pooling only for first layer to bring to 3x5
            nonlin = [nonlin_type, nonlin_type, nonlin_type, 'none'] # No non linearity for last layer
            norm   = [norm_type, norm_type, norm_type, 'none'] # No batch norm for last layer
            self.outencoder = nn.Sequential(
                *[ConvType(chn_out[k], chn_out[k+1], kernel_size=1, stride=1, padding=0,
                           use_pool=pool[k], norm_type=norm[k], nonlinearity=nonlin[k]) for k in range(len(chn_out)-1)],
            )
            self.outdim = [chn_out[-1], 3, 5] if self.deterministic else [chn_out[-1]//2, 3, 5]
        else:
            # Fully connected output network
            print("[Encoder] Using fully-connected network for state prediction")
            odim = [sdim[-1]+(chn[-1]*7*10), 512, 512, 512] if wide_model else [sdim[-1]+(chn[-1]*7*10), 256, 256, 256]
            self.outencoder = nn.Sequential(
                nn.Linear(odim[0], odim[1]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(odim[1], odim[2]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(odim[2], odim[3]),
            )
            self.outdim = [odim[3]] if self.deterministic else [odim[3]//2]

        # Deterministic model
        self.deterministic = deterministic
        if deterministic:
            print("[Encoder] Prediciting deterministic output state")

    def forward(self, img, state=None):
        # Encode the image data
        bsz = img.size(0)
        h_img = self.imgencoder(img)

        # Encode state information
        h_state = None
        if (state is not None) and self.use_state:
            h_state = self.stateencoder(state)

        # Concat outputs and get final output
        if self.conv_encode:
            # Convert the state encoding to an image representation (if present)
            if (h_state is not None):
                # Reshape the output of the state encoder (repeat along image dims)
                _, ndim = h_state.size()
                _, _, ht, wd = h_img.size()
                h_state = h_state.view(bsz, ndim, 1, 1).expand(bsz, ndim, ht, wd)

                # Concat along channels dimension
                h_both = torch.cat([h_img, h_state], 1)
            else:
                h_both = h_img
        else:
            # Run through fully connected encoders
            h_img  = h_img.view(h_img.size(0), -1) # Flatten output

            # Concat state output if present
            if (h_state is not None):
                h_both = torch.cat([h_img, h_state], 1)
            else:
                h_both = h_img

        # Get output
        h_out = self.outencoder(h_both)

        # Distributional vs Deterministic output
        if self.deterministic:
            return h_out # Directly return predicted hidden state
        else:
            # Get mean & log(std) output
            mean, logstd = h_out.split(h_out.size(1)//2, dim=1) # Split into 2 along channels dim (or hidden state dim)

            # Create a independent diagonal Gaussian
            return Normal(loc=mean.view(bsz,-1), scale=torch.exp(logstd).view(bsz,-1))

        # Create the co-variance matrix (as a diagonal matrix with predicted stds along the diagonal)
        #var    = torch.exp(logstd).pow(2).view(bsz,-1)
        #covar  = torch.stack([torch.diag(var[k]) for k in range(bsz)], 0) # B x N x N matrix

        # Return a Multi-variate normal distribution
        #return MVNormal(loc=mean.view(bsz,-1), covariance_matrix=covar)

#############################
##### Create Decoder
class Decoder(nn.Module):
    def __init__(self, img_type='rgbd', norm_type='bn', nonlin_type='prelu', wide_model=False,
                 pred_state=False, num_state=7, coord_conv=False,
                 conv_decode=False, rgb_normalize=False):
        super(Decoder, self).__init__()

        # Normalization
        assert (norm_type in ['none', 'bn', 'ln', 'in']), "Unknown normalization type input: {}".format(norm_type)

        # Coordinate convolution
        if coord_conv:
            print('[Decoder] Using co-ordinate convolutional layers')
        ConvType   = lambda x, y, **v: BasicConv2D(in_channels=x, out_channels=y,
                                                   coord_conv=coord_conv, **v)
        DeconvType = lambda x, y, **v: BasicDeconv2D(in_channels=x, out_channels=y,
                                                     coord_conv=coord_conv, **v)

        # Output type
        print("[Decoder] Output image type: {}, Normalization type: {}, Nonliearity: {}".format(
            img_type, norm_type, nonlin_type))
        self.img_type, self.rgb_normalize = img_type, rgb_normalize
        if img_type == 'rgb' or img_type == 'xyz':
            output_channels = 3
        elif img_type == 'd':
            output_channels = 1
        elif img_type == 'rgbd':
            output_channels = 4
        elif img_type == 'rgbxyz':
            output_channels = 6
        else:
            assert False, "Unknown image type input: {}".format(img_type)

        # RGB normalization
        if self.rgb_normalize:
            print('[Decoder] RGB output from the decoder is normalized to go from 0-1')

        ###### Decode XYZ-RGB images (half-size since we are using half size of inputs compared to before)
        # Create conv-decoder (large net => 5 conv layers with pooling)
        # 3x4, 7x10 -> 15x20
        # 4x4, 15x20 -> 30x40
        # 6x6, 30x40 -> 60x80
        # 6x6, 60x80 -> 120x160
        # 8x8, 120x160 -> 240x320
        chn = [128, 128, 64, 32, 32, output_channels] if wide_model else [64, 64, 32, 16, 16, output_channels]  # Num channels
        kern = [(3,4), 4, 6, 6, 8]  # Kernel sizes
        padd = [(0,1), 1, 2, 2, 3]  # Padding
        norm = [norm_type, norm_type, norm_type, norm_type, 'none'] # No batch norm for last layer (output)
        nonlin = [nonlin_type, nonlin_type, nonlin_type, nonlin_type, 'none'] # No non-linearity last layer (output)
        self.idecdim = [chn[0], 7, 10]
        self.imgdecoder = nn.Sequential(
            *[DeconvType(chn[k], chn[k+1], kernel_size=kern[k], stride=2, padding=padd[k],
                         norm_type= norm[k], nonlinearity=nonlin[k])
              for k in range(len(chn)-1)]
        )

        # ###### Decode state information (jt angles, gripper position)
        ## todo: Add decoding of state
        assert(not pred_state)
        # self.pred_state = pred_state
        # if self.pred_state:
        #     print('[Decoder] Predicting state output using the decoder')
        #     sdim = [self.state_chn*self.indim[1]*self.indim[2], 64, 32, num_state]
        #     self.statedecoder = nn.Sequential(
        #         nn.Linear(sdim[0], sdim[1]),
        #         ctrlnets.get_nonlinearity(nonlin_type),
        #         nn.Linear(sdim[1], sdim[2]),
        #         ctrlnets.get_nonlinearity(nonlin_type),
        #         nn.Linear(sdim[2], sdim[3]),
        #     )
        # else:
        #     sdim = [0]
        # self.state_chn = 4 if pred_state else 0 # Add a few extra channels if we are also predicting state

        ###### If we are using conv output (fully-conv basically), we can do 1x1 conv stuff
        self.conv_decode = conv_decode
        if self.conv_decode:
            # 1x1, 3x5 -> 3x5, 16 -> 32
            # 1x1, 3x5 -> 3x5, 32 -> 64
            # 1x1, 3x5 -> 3x5, 64 -> 128
            # 1x1, 3x5 -> 7x10, 128 -> 128 + 4 (4 extra channels are inputs to the joint encoder)
            print("[Decoder] Using convolutional network with 1x1 convolutions for predicting output image")
            chn_in = [16, 32, 64, 128, chn[0]] if wide_model else [8, 16, 32, 64, chn[0]]
            self.hsdecoder = nn.Sequential(
                *[ConvType(chn_in[k], chn_in[k+1], kernel_size=1, stride=1, padding=0,
                           use_pool=False, norm_type=norm_type, nonlinearity=nonlin_type) for k in range(len(chn_in)-2)], # Leave out last layer
                nn.UpsamplingBilinear2d((7, 10)), # Upsample to (7,10)
                ConvType(chn_in[3], chn_in[4], kernel_size=1, stride=1, padding=0,
                         use_pool=False, norm_type=norm_type, nonlinearity=nonlin_type), # 128 x 7 x 10
            )
            self.hsdim = [chn_in[0], 3, 5]
        else:
            # Fully connected output network
            print("[Decoder] Using fully-connected network for predicting output image")
            odim = [256, 256, 256, chn[0]*7*10] if wide_model else [128, 128, 128, chn[0]*7*10]
            self.hsdecoder = nn.Sequential(
                nn.Linear(odim[0], odim[1]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(odim[1], odim[2]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(odim[2], odim[3]),
            )
            self.hsdim = [odim[0]]

    def forward(self, hidden_state):
        # Reshape hidden state to right shape
        bsz = hidden_state.size(0)
        h_state = hidden_state.view(bsz, *self.hsdim)

        # Pass it through the indecoder
        h_img = self.hsdecoder(h_state).view(bsz, *self.idecdim)

        # Pass it through the image decoder
        imgout = self.imgdecoder(h_img)

        # If we have RGB images & are asked to normalize the output, push it through a sigmoid to get vals from 0->1
        if (self.img_type.find('rgb') != -1) and self.rgb_normalize:
            splits = imgout.split(3, dim=1) # RGB is first element
            rgb    = F.sigmoid(splits[0]) # First 3 channels
            output = torch.cat([rgb, *splits[1:]], 1) if len(splits) > 1 else rgb
        else:
            output = imgout

        # Return
        return output

#############################
##### Create Transition Model
# types:
# 1) sample -> delta/full sample
# 2) sample -> delta/full mean, delta/full var
# 3) mean, var -> delta/full mean, delta/full var
class NonLinearTransitionModel(nn.Module):
    def __init__(self, state_dim, ctrl_dim, setting='dist2dist',
                 predict_deltas=False, wide_model=False, nonlin_type='prelu',
                 conv_net=False, norm_type='none', coord_conv=False):
        super(NonLinearTransitionModel, self).__init__()

        ### Allowed settings
        self.setting = setting
        assert (setting in ['samp2samp', 'samp2dist', 'dist2dist']), \
            "Unknown setting {} for the transition model".format(setting)
        assert(type(state_dim) == list)
        if setting == 'samp2samp':
            in_dim, out_dim = state_dim.copy(), state_dim.copy()
        elif setting == 'samp2dist':
            in_dim, out_dim = state_dim.copy(), state_dim.copy()
            out_dim[0] *= 2 # Predict both mean/var
        else: #setting == 'dist2dist':
            in_dim, out_dim = state_dim.copy(), state_dim.copy()
            in_dim[0]  *= 2
            out_dim[0] *= 2
        print('[Transition] Setting: {}, Nonlinearity: {}, Normalization type: {}'.format(
            setting, nonlin_type, norm_type))

        # Setup encoder for ctrl input
        cdim = [ctrl_dim, 64, 128, 128] if wide_model else [ctrl_dim, 32, 64, 64]
        self.ctrlencoder = nn.Sequential(
            nn.Linear(cdim[0], cdim[1]),
            ctrlnets.get_nonlinearity(nonlin_type),
            nn.Linear(cdim[1], cdim[2]),
            ctrlnets.get_nonlinearity(nonlin_type),
            nn.Linear(cdim[2], cdim[3])
        )

        # Coordinate convolution
        if coord_conv:
            print('[Transition] Using co-ordinate convolutional layers')
        ConvType = lambda x, y, **v: BasicConv2D(in_channels=x, out_channels=y,
                                                 coord_conv=coord_conv, **v)

        # Setup state encoder and decoder (conv vs FC)
        self.conv_net       = conv_net
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.predict_deltas = predict_deltas
        if conv_net:
            # Normalization
            print('[Transition] Using fully-convolutional transition network')
            assert (norm_type in ['none', 'bn', 'ln', 'in']), "Unknown normalization type input: {}".format(norm_type)

            # Create 1x1 conv state encoder
            sedim = [in_dim[0], 32, 64, 128] if wide_model else [in_dim[0], 16, 32, 64]
            self.stateencoder = nn.Sequential(
                *[ConvType(sedim[k], sedim[k+1], kernel_size=1, stride=1, padding=0,
                           use_pool=False, norm_type=norm_type, nonlinearity=nonlin_type) for k in range(len(sedim)-1)],
            )

            # Create 1x1 conv state decoder (use dimensions of control encoding as channels, replicate values in height & width)
            sddim = [sedim[-1]+cdim[-1], 128, 64, 32, out_dim[0]] if wide_model else [sedim[-1]+cdim[-1], 64, 32, 16, out_dim[0]]
            nonlin = [nonlin_type, nonlin_type, nonlin_type, 'none']  # No non linearity for last layer
            norm   = [norm_type, norm_type, norm_type, False]  # No batch norm for last layer
            self.statedecoder = nn.Sequential(
                *[ConvType(sddim[k], sddim[k+1], kernel_size=1, stride=1, padding=0,
                           use_pool=False, norm_type=norm[k], nonlinearity=nonlin[k]) for k in range(len(sddim) - 1)],
            )
        else:
            # Create FC state encoder
            print('[Transition] Using fully-connected transition network')
            sedim = [in_dim[0], 512, 512, 512] if wide_model else [in_dim[0], 256, 256, 256]
            self.stateencoder = nn.Sequential(
                nn.Linear(sedim[0], sedim[1]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sedim[1], sedim[2]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sedim[2], sedim[3]),
            )

            # Create FC state decoder
            sddim = [sedim[-1]+cdim[-1], 512, 512, out_dim[0]] if wide_model else [sedim[-1]+cdim[-1], 256, 256, out_dim[0]]
            self.statedecoder = nn.Sequential(
                nn.Linear(sddim[0], sddim[1]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sddim[1], sddim[2]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sddim[2], sddim[3]),
            )

        # Prints
        if predict_deltas:
            print('[Transition] Model predicts deltas, not full state')

    def forward(self, ctrl, inpsample=None, inpdist=None):
        # Generate the state input based on the setting
        if (self.setting == 'samp2samp'):
            assert(inpsample is not None)
            bsz   = inpsample.size(0)
            state = inpsample.view(bsz, *self.in_dim)
            mean, var = None, None
        elif (self.setting == 'samp2dist'):
            assert((inpsample is not None) and (inpdist is not None))
            bsz       = inpsample.size(0)
            state     = inpsample.view(bsz, *self.in_dim) # Use sample as state
            mean, std = inpdist.mean, inpdist.stddev
            #mean, var = inpdist.mean, torch.stack([torch.diag(inpdist.covariance_matrix[k]) for k in range(bsz)], 0)
        else:
            assert(inpdist is not None)
            # Get the mean & variance from the input distribution
            bsz       = inpdist.mean.size(0)
            mean, std = inpdist.mean, inpdist.stddev
            #mean, var = inpdist.mean, torch.stack([torch.diag(inpdist.covariance_matrix[k]) for k in range(bsz)], 0)
            state     = torch.cat([mean, std], 0).view(bsz, *self.in_dim)

        # Run control through encoder
        h_ctrl  = self.ctrlencoder(ctrl)

        # Run state through encoder
        h_state = self.stateencoder(state)

        # Based on conv net vs not, reshape hidden state of the ctrl encoder as a 4D tensor
        if self.conv_net:
            # Reshape the output of the ctrl encoder as an image (repeat along image dims)
            _, ndim      = h_ctrl.size()
            _, _, ht, wd = h_state.size()
            h_ctrl       = h_ctrl.view(bsz, ndim, 1, 1).expand(bsz, ndim, ht, wd)

        # Concat hidden states along channels dimension
        h_both = torch.cat([h_state, h_ctrl], 1)

        # Generate decoded output from state decoder (either deltas or full output)
        h_out = self.statedecoder(h_both)

        # Run through state decoder to predict output, return output sample/dist (if asked)
        if (self.setting == 'samp2samp'):
            # Since it is a sample, we can just add the deltas (no need to worry about -ves)
            if self.predict_deltas:
                nextstate = (state + h_out)
            else:
                nextstate = h_out

            # Return
            return nextstate.view_as(inpsample)
        else:
            # Predict distributions in other two cases
            pred_mean, pred_logstd = h_out.split(h_out.size(1)//2, dim=1)
            pred_std = torch.exp(pred_logstd)
            #pred_var = torch.exp(pred_logstd).pow(2)

            # Add deltas in mean/std space (if predicting deltas)
            if self.predict_deltas:
                next_mean, next_std   = mean + pred_mean.view(bsz,-1), \
                                        std + pred_std.view(bsz,-1)
            else:
                next_mean, next_std   = pred_mean.view(bsz,-1), \
                                        pred_std.view(bsz,-1)

            # Return distribution
            return Normal(loc=next_mean, scale=next_std)

            # # Add deltas in mean/var space (if predicting deltas)
            # if self.predict_deltas:
            #     next_mean, next_var = mean + pred_mean.view(bsz,-1), \
            #                           var + pred_var.view(bsz,-1)
            # else:
            #     next_mean, next_var = pred_mean.view(bsz,-1), \
            #                           pred_var.view(bsz,-1)
            #
            # # Create the covariance matrix
            # next_covar = torch.stack([torch.diag(next_var[k]) for k in range(bsz)], 0) # B x N x N matrix
            #
            # # Return distribution
            # return MVNormal(loc=next_mean, covariance_matrix=next_covar)


# todo: locally linear transition model

##########################################################
##### Create E2C model
class E2CModel(nn.Module):
    def __init__(self, enc_img_type='rgbd', dec_img_type='rgbd',
                 enc_inp_state=True, dec_pred_state=False,
                 conv_enc_dec=True, dec_pred_norm_rgb=True,
                 trans_setting='dist2dist', trans_pred_deltas=True,
                 trans_model_type='nonlin', state_dim=8, ctrl_dim=8,
                 wide_model=False, nonlin_type='prelu',
                 norm_type='bn', coord_conv=False, img_size=(240,320)):
        super(E2CModel, self).__init__()

        # Use different encoder/decoder/trans model functions for conv_enc_dec
        if conv_enc_dec:
            # Setup encoder
            self.encoder = ConvEncoder(img_type=enc_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                       use_state=enc_inp_state, num_state=state_dim,
                                       coord_conv=coord_conv, img_size=img_size)

            # Setup decoder
            self.decoder = ConvDecoder(img_type=dec_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                       pred_state=dec_pred_state, num_state=state_dim,
                                       coord_conv=coord_conv, rgb_normalize=dec_pred_norm_rgb,
                                       img_size=img_size, input_size=self.encoder.outdim)

            # Setup transition model
            self.transition = ConvTransitionModel(state_dim=self.encoder.outdim, ctrl_dim=ctrl_dim,
                                                  setting=trans_setting, predict_deltas=trans_pred_deltas,
                                                  nonlin_type=nonlin_type, norm_type=norm_type,
                                                  coord_conv=coord_conv)
        else:
            # Setup encoder
            self.encoder = Encoder(img_type=enc_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                   wide_model=wide_model, use_state=enc_inp_state, num_state=state_dim,
                                   coord_conv=coord_conv, conv_encode=False)

            # Setup decoder
            self.decoder = Decoder(img_type=dec_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                   wide_model=wide_model, pred_state=dec_pred_state, num_state=state_dim,
                                   coord_conv=coord_conv, conv_decode=False, rgb_normalize=dec_pred_norm_rgb)

            # Setup transition model
            if trans_model_type == 'nonlin':
                print("[Transition] Using non-linear transition model")
                self.transition = NonLinearTransitionModel(state_dim=self.encoder.outdim, ctrl_dim=ctrl_dim,
                                                           setting=trans_setting, predict_deltas=trans_pred_deltas,
                                                           wide_model=wide_model, nonlin_type=nonlin_type,
                                                           conv_net=False, norm_type=norm_type,
                                                           coord_conv=coord_conv)
            else:
                assert False, "Unknown transition model type: {}".format(trans_model_type)

    # Inputs are (B x (S+1) x C x H x W), (B x (S+1) x NDIM), (B x S x NDIM)
    # Outputs are lists of length (S+1) or (S) with dimensions being (B x NDIM) or (B x C x H x W)
    def forward(self, imgs, states, ctrls):
        # Encode the images and states through the encoder
        encdists, encsamples = [], []
        for k in range(imgs.size(1)): # imgs is B x (S+1) x C x H x W
            dist = self.encoder.forward(imgs[:,k], states[:,k]) # Predict the hidden state distribution
            samp = dist.rsample_e2c() # Generate a sample from the predicted distribution

            # Save the predictions
            encdists.append(dist)
            encsamples.append(samp)

        # Predict a sequence of next states using the transition model
        transdists, transsamples = [], []
        setting = self.transition.setting
        for k in range(ctrls.size(1)): # ctrls is B x S x NCTRL
            # Get the inputs
            transinpsample = encsamples[0] if (k == 0) else transsamples[-1] # @t=0 use encoder input, else use own prediction
            transinpdist   = encdists[0] if (k == 0) else transdists[-1] # @t=0 use encoder input, else use own prediction

            # Make a prediction through the transition model
            transout = self.transition.forward(ctrls[:,k], transinpsample, transinpdist)

            # Get the output distribution & sample
            if (setting.find('2dist') != -1): # Predicting a distributional output
                transdists.append(transout)
                transsamples.append(transout.rsample_e2c()) # Sample from next state distribution
            else:
                transdists.append(None) # We are predicting a sample, no distribution here
                transsamples.append(transout)

        # Decode the frames based on the encoder and transition model
        # Encoder sample is used to generate frame @ t = 0, transition model samples are used for t = 1 to N
        # todo: do we need to reconstruct from all encoded state samples too?
        decimgs = []
        for k in range(imgs.size(1)):
            decinp = encsamples[0] if (k == 0) else transsamples[k-1] # @ t = 0 reconstruct from encoder, rest from transition
            decimg = self.decoder.forward(decinp)
            decimgs.append(decimg)

        # Return stuff
        return encdists, encsamples, transdists, transsamples, decimgs

#######################################################################################
##### Create Convolutional encoder
class ConvEncoder(nn.Module):
    def __init__(self, img_type='rgbd', norm_type='bn', nonlin_type='prelu',
                 use_state=False, num_state=7, coord_conv=False,
                 img_size=(240,320), deterministic=False):
        super(ConvEncoder, self).__init__()

        # Normalization
        assert (norm_type in ['none', 'bn', 'ln', 'in']), "Unknown normalization type input: {}".format(norm_type)

        # Coordinate convolution
        if coord_conv:
            print('[Encoder] Using co-ordinate convolutional layers')
        ConvType = lambda x, y, **v: BasicConv2D(in_channels=x, out_channels=y,
                                                 coord_conv=coord_conv, **v)

        # Input type
        print("[Encoder] Input image type: {}, Normalization type: {}, Nonliearity: {}".format(
            img_type, norm_type, nonlin_type))
        if img_type == 'rgb' or img_type == 'xyz':
            input_channels = 3
        elif img_type == 'd':
            input_channels = 1
        elif img_type == 'rgbd':
            input_channels = 4
        elif img_type == 'rgbxyz':
            input_channels = 6
        else:
            assert False, "Unknown image type input: {}".format(img_type)

        ###### Encode XYZ-RGB images
        # Create conv-encoder, stride 1, 7x7/5x5 convs with max pooling & BN
        # 4/5-conv layers, final output is size: 128x8x8
        self.img_size = img_size
        if self.img_size == (240,320):
            print('[Encoder] Image size: {}, Using 5-conv layers'.format(self.img_size))
            chn  = [input_channels, 32, 64, 64, 128, 128] # Num channels
            kern = [7,7,5,(4,5),(4,5)] # Kernel sizes
            pad  = [3,3,2,(2,1),(2,1)] # Padding
        elif self.img_size == (128,128):
            print('[Encoder] Image size: {}, Using 4-conv layers'.format(self.img_size))
            chn  = [input_channels, 32, 64, 64, 128]  # Num channels
            kern = [7,7,5,5] # Kernel sizes
            pad  = [3,3,2,2] # Padding
        else:
            assert False, "Unknown image size input: {}".format(self.img_size)
        self.imgencoder = nn.Sequential(
            *[ConvType(chn[k], chn[k+1], kernel_size=kern[k], stride=1, padding=pad[k],
                      use_pool=True, norm_type=norm_type, nonlinearity=nonlin_type) for k in range(len(chn)-1)]
        )

        ###### Encode state information (jt angles, gripper position)
        self.use_state = use_state
        if self.use_state:
            print("[Encoder] Using state as input")
            sdim = [num_state, 32, 64]
            self.stateencoder = nn.Sequential(
                nn.Linear(sdim[0], sdim[1]),
                ctrlnets.get_nonlinearity(nonlin_type),
                nn.Linear(sdim[1], sdim[2])
            )
        else:
            sdim = [0]

        ###### Concat img + state information and get encoded outputs
        # Convolutional network with 5x5 convolutions to get some global information
        print("[Encoder] Using convolutional network for state prediction")
        out_chn = 64 if deterministic else 128 # Predict mean/var if not deterministic
        chn_out = [sdim[-1]+chn[-1], 128, out_chn]
        nonlin  = [nonlin_type, 'none'] # No non linearity for last layer
        norm    = [norm_type, 'none']       # No batch norm for last layer
        self.outencoder = nn.Sequential(
            *[ConvType(chn_out[k], chn_out[k+1], kernel_size=5, stride=1, padding=2,
                       use_pool=False, norm_type=norm[k], nonlinearity=nonlin[k]) for k in range(len(chn_out)-1)],
        )
        self.outdim = [chn_out[-1], 8, 8] if deterministic else [chn_out[-1]//2, 8, 8]

        # Deterministic model
        self.deterministic = deterministic
        if deterministic:
            print("[Encoder] Prediciting deterministic output state")

    def forward(self, img, state=None):
        # Encode the image data
        bsz = img.size(0)
        h_img = self.imgencoder(img)

        # Encode state information
        h_state = None
        if (state is not None) and self.use_state:
            h_state = self.stateencoder(state)

        # Convert the state encoding to an image representation (if present)
        if (h_state is not None):
            # Reshape the output of the state encoder (repeat along image dims)
            _, ndim = h_state.size()
            _, _, ht, wd = h_img.size()
            h_state = h_state.view(bsz, ndim, 1, 1).expand(bsz, ndim, ht, wd)

            # Concat along channels dimension
            h_both = torch.cat([h_img, h_state], 1)
        else:
            h_both = h_img

        # Get output
        h_out = self.outencoder(h_both)

        # Distributional vs Deterministic output
        if self.deterministic:
            return h_out # Directly return predicted hidden state
        else:
            # Get mean & log(std) output
            mean, logstd = h_out.split(h_out.size(1)//2, dim=1) # Split into 2 along channels dim (or hidden state dim)

            # Create a independent diagonal Gaussian
            return Normal(loc=mean.view(bsz,-1), scale=torch.exp(logstd).view(bsz,-1))

        # Create the co-variance matrix (as a diagonal matrix with predicted stds along the diagonal)
        #var    = torch.exp(logstd).pow(2).view(bsz,-1)
        #covar  = torch.stack([torch.diag(var[k]) for k in range(bsz)], 0) # B x N x N matrix

        # Return a Multi-variate normal distribution
        #return MVNormal(loc=mean.view(bsz,-1), covariance_matrix=covar)

#############################
##### Create Decoder
class ConvDecoder(nn.Module):
    def __init__(self, img_type='rgbd', norm_type='bn', nonlin_type='prelu',
                 pred_state=False, num_state=7, coord_conv=False,
                 rgb_normalize=False, input_size=(64,8,8), img_size=(240,320)):
        super(ConvDecoder, self).__init__()

        # Normalization
        assert (norm_type in ['none', 'bn', 'ln', 'in']), "Unknown normalization type input: {}".format(norm_type)

        # Coordinate convolution
        if coord_conv:
            print('[Decoder] Using co-ordinate convolutional layers')
        DeconvType = lambda x, y, **v: BasicDeconv2D(in_channels=x, out_channels=y,
                                                     coord_conv=coord_conv, **v)

        # Output type
        print("[Decoder] Output image type: {}, Normalization type: {}, Nonliearity: {}".format(
            img_type, norm_type, nonlin_type))
        self.img_type, self.rgb_normalize = img_type, rgb_normalize
        if img_type == 'rgb' or img_type == 'xyz':
            output_channels = 3
        elif img_type == 'd':
            output_channels = 1
        elif img_type == 'rgbd':
            output_channels = 4
        elif img_type == 'rgbxyz':
            output_channels = 6
        else:
            assert False, "Unknown image type input: {}".format(img_type)

        # RGB normalization
        if self.rgb_normalize:
            print('[Decoder] RGB output from the decoder is normalized to go from 0-1')

        ###### Decode XYZ-RGB images
        # Create conv-decoder, 7x7/5x5 strided deconvs with BN
        # 4/5-conv layers, initial input is size: 64x8x8, output is image size
        self.img_size = img_size
        self.inp_size = input_size
        if self.img_size == (240,320):
            print('[Decoder] Image size: {}, Using 5-deconv layers'.format(self.img_size))
            chn = [input_size[0], 64, 64, 32, 32, output_channels] # Num channels
            kern = [(3,6), (4,6), 6, 6, 6] # Kernel sizes
            padd = [1, (1,0), 2, 2, 2] # Padding
            norm = [norm_type, norm_type, norm_type, norm_type, 'none']  # No batch norm for last layer (output)
            nonlin = [nonlin_type, nonlin_type, nonlin_type, nonlin_type, 'none']  # No non-linearity last layer
        elif self.img_size == (128,128):
            print('[Decoder] Image size: {}, Using 4-deconv layers'.format(self.img_size))
            chn = [input_size[0], 64, 64, 32, output_channels] # Num channels
            kern = [4, 4, 6, 6] # Kernel sizes
            padd = [1, 1, 2, 2] # Padding
            norm = [norm_type, norm_type, norm_type, 'none']  # No batch norm for last layer (output)
            nonlin = [nonlin_type, nonlin_type, nonlin_type, 'none']  # No non-linearity last layer
        else:
            assert False, "Unknown image size input: {}".format(self.img_size)
        self.imgdecoder = nn.Sequential(
            *[DeconvType(chn[k], chn[k+1], kernel_size=kern[k], stride=2, padding=padd[k],
                         norm_type= norm[k], nonlinearity=nonlin[k])
              for k in range(len(chn)-1)]
        )

        # ###### Decode state information (jt angles, gripper position)
        ## todo: Add decoding of state - 1/2 conv layers to get to small num values & then FC
        assert(not pred_state)
        # self.pred_state = pred_state
        # if self.pred_state:
        #     print('[Decoder] Predicting state output using the decoder')
        #     sdim = [self.state_chn*self.indim[1]*self.indim[2], 64, 32, num_state]
        #     self.statedecoder = nn.Sequential(
        #         nn.Linear(sdim[0], sdim[1]),
        #         ctrlnets.get_nonlinearity(nonlin_type),
        #         nn.Linear(sdim[1], sdim[2]),
        #         ctrlnets.get_nonlinearity(nonlin_type),
        #         nn.Linear(sdim[2], sdim[3]),
        #     )
        # else:
        #     sdim = [0]
        # self.state_chn = 4 if pred_state else 0 # Add a few extra channels if we are also predicting state

    def forward(self, hidden_state):
        # Reshape hidden state to right shape
        bsz = hidden_state.size(0)
        h_state = hidden_state.view(bsz, *self.inp_size)

        # Pass it through the image decoder
        imgout = self.imgdecoder(h_state)

        # If we have RGB images & are asked to normalize the output, push it through a sigmoid to get vals from 0->1
        if (self.img_type.find('rgb') != -1) and self.rgb_normalize:
            splits = imgout.split(3, dim=1) # RGB is first element
            rgb    = F.sigmoid(splits[0]) # First 3 channels
            output = torch.cat([rgb, *splits[1:]], 1) if len(splits) > 1 else rgb
        else:
            output = imgout

        # Return
        return output

#############################
##### Create Transition Model
# types:
# 1) sample -> delta/full sample
# 2) sample -> delta/full mean, delta/full var
# 3) mean, var -> delta/full mean, delta/full var
class ConvTransitionModel(nn.Module):
    def __init__(self, state_dim, ctrl_dim, setting='dist2dist',
                 predict_deltas=False, nonlin_type='prelu',
                 norm_type='none', coord_conv=False):
        super(ConvTransitionModel, self).__init__()

        ### Allowed settings
        self.setting = setting
        assert (setting in ['samp2samp', 'samp2dist', 'dist2dist']), \
            "Unknown setting {} for the transition model".format(setting)
        assert(type(state_dim) == list)
        if setting == 'samp2samp':
            in_dim, out_dim = state_dim.copy(), state_dim.copy()
        elif setting == 'samp2dist':
            in_dim, out_dim = state_dim.copy(), state_dim.copy()
            out_dim[0] *= 2 # Predict both mean/var
        else: #setting == 'dist2dist':
            in_dim, out_dim = state_dim.copy(), state_dim.copy()
            in_dim[0]  *= 2
            out_dim[0] *= 2
        print('[Transition] Setting: {}, Nonlinearity: {}, Normalization type: {}'.format(
            setting, nonlin_type, norm_type))

        # Setup encoder for ctrl input
        cdim = [ctrl_dim, 32, 64]
        self.ctrlencoder = nn.Sequential(
            nn.Linear(cdim[0], cdim[1]),
            ctrlnets.get_nonlinearity(nonlin_type),
            nn.Linear(cdim[1], cdim[2]),
        )

        # Coordinate convolution
        if coord_conv:
            print('[Transition] Using co-ordinate convolutional layers')
        ConvType = lambda x, y, **v: BasicConv2D(in_channels=x, out_channels=y,
                                                 coord_conv=coord_conv, **v)

        # Setup state encoder and decoder (conv vs FC)
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.predict_deltas = predict_deltas

        # Normalization
        print('[Transition] Using convolutional transition network')
        assert (norm_type in ['none', 'bn', 'ln', 'in']), "Unknown normalization type input: {}".format(norm_type)

        # Create 1x1 conv state encoder
        sedim = [in_dim[0], 128, 128]
        nonlin = [nonlin_type, 'none']
        norm   = [norm_type, 'none']
        self.stateencoder = nn.Sequential(
            *[ConvType(sedim[k], sedim[k+1], kernel_size=5, stride=1, padding=2,
                       use_pool=False, norm_type=norm[k], nonlinearity=nonlin[k]) for k in range(len(sedim)-1)],
        )

        # Create 1x1 conv state decoder (use dimensions of control encoding as channels, replicate values in height & width)
        sddim = [sedim[-1]+cdim[-1], 128, 128, out_dim[0]]
        nonlin = [nonlin_type, nonlin_type, 'none']  # No non linearity for last layer
        norm   = [norm_type, norm_type, 'none']  # No batch norm for last layer
        self.statedecoder = nn.Sequential(
            *[ConvType(sddim[k], sddim[k+1], kernel_size=5, stride=1, padding=2,
                       use_pool=False, norm_type=norm[k], nonlinearity=nonlin[k]) for k in range(len(sddim) - 1)],
        )

        # Prints
        if predict_deltas:
            print('[Transition] Model predicts deltas, not full state')

    def forward(self, ctrl, inpsample=None, inpdist=None):
        # Generate the state input based on the setting
        if (self.setting == 'samp2samp'):
            assert(inpsample is not None)
            bsz   = inpsample.size(0)
            state = inpsample.view(bsz, *self.in_dim)
            mean, var = None, None
        elif (self.setting == 'samp2dist'):
            assert((inpsample is not None) and (inpdist is not None))
            bsz       = inpsample.size(0)
            state     = inpsample.view(bsz, *self.in_dim) # Use sample as state
            mean, std = inpdist.mean, inpdist.stddev
            #mean, var = inpdist.mean, torch.stack([torch.diag(inpdist.covariance_matrix[k]) for k in range(bsz)], 0)
        else:
            assert(inpdist is not None)
            # Get the mean & variance from the input distribution
            bsz       = inpdist.mean.size(0)
            mean, std = inpdist.mean, inpdist.stddev
            #mean, var = inpdist.mean, torch.stack([torch.diag(inpdist.covariance_matrix[k]) for k in range(bsz)], 0)
            state     = torch.cat([mean, std], 0).view(bsz, *self.in_dim)

        # Run control through encoder
        h_ctrl  = self.ctrlencoder(ctrl)

        # Run state through encoder
        h_state = self.stateencoder(state)

        # Reshape the output of the ctrl encoder as an image (repeat along image dims)
        _, ndim      = h_ctrl.size()
        _, _, ht, wd = h_state.size()
        h_ctrl       = h_ctrl.view(bsz, ndim, 1, 1).expand(bsz, ndim, ht, wd)

        # Concat hidden states along channels dimension
        h_both = torch.cat([h_state, h_ctrl], 1)

        # Generate decoded output from state decoder (either deltas or full output)
        h_out = self.statedecoder(h_both)

        # Run through state decoder to predict output, return output sample/dist (if asked)
        if (self.setting == 'samp2samp'):
            # Since it is a sample, we can just add the deltas (no need to worry about -ves)
            if self.predict_deltas:
                nextstate = (state + h_out)
            else:
                nextstate = h_out

            # Return
            return nextstate.view_as(inpsample)
        else:
            # Predict distributions in other two cases
            pred_mean, pred_logstd = h_out.split(h_out.size(1)//2, dim=1)
            pred_std = torch.exp(pred_logstd)
            #pred_var = torch.exp(pred_logstd).pow(2)

            # Add deltas in mean/std space (if predicting deltas)
            if self.predict_deltas:
                next_mean, next_std   = mean + pred_mean.view(bsz,-1), \
                                        std + pred_std.view(bsz,-1)
            else:
                next_mean, next_std   = pred_mean.view(bsz,-1), \
                                        pred_std.view(bsz,-1)

            # Return distribution
            return Normal(loc=next_mean, scale=next_std)

            # # Add deltas in mean/var space (if predicting deltas)
            # if self.predict_deltas:
            #     next_mean, next_var = mean + pred_mean.view(bsz,-1), \
            #                           var + pred_var.view(bsz,-1)
            # else:
            #     next_mean, next_var = pred_mean.view(bsz,-1), \
            #                           pred_var.view(bsz,-1)
            #
            # # Create the covariance matrix
            # next_covar = torch.stack([torch.diag(next_var[k]) for k in range(bsz)], 0) # B x N x N matrix
            #
            # # Return distribution
            # return MVNormal(loc=next_mean, covariance_matrix=next_covar)

##########################################################
##### Create E2C model
class DeterministicModel(nn.Module):
    def __init__(self, enc_img_type='rgbd', dec_img_type='rgbd',
                 enc_inp_state=True, dec_pred_state=False,
                 conv_enc_dec=True, dec_pred_norm_rgb=True,
                 trans_setting='samp2samp', trans_pred_deltas=True,
                 trans_model_type='nonlin', state_dim=8, ctrl_dim=8,
                 wide_model=False, nonlin_type='prelu',
                 norm_type='bn', coord_conv=False, img_size=(240,320)):
        super(DeterministicModel, self).__init__()

        # Use different encoder/decoder/trans model functions for conv_enc_dec
        assert (trans_setting == 'samp2samp'), "Deterministic model only allows samp2samp transition model setting"
        if conv_enc_dec:
            # Setup encoder
            self.encoder = ConvEncoder(img_type=enc_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                       use_state=enc_inp_state, num_state=state_dim,
                                       coord_conv=coord_conv, img_size=img_size, deterministic=True)

            # Setup decoder
            self.decoder = ConvDecoder(img_type=dec_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                       pred_state=dec_pred_state, num_state=state_dim,
                                       coord_conv=coord_conv, rgb_normalize=dec_pred_norm_rgb,
                                       img_size=img_size, input_size=self.encoder.outdim)

            # Setup transition model
            self.transition = ConvTransitionModel(state_dim=self.encoder.outdim, ctrl_dim=ctrl_dim,
                                                  setting=trans_setting, predict_deltas=trans_pred_deltas,
                                                  nonlin_type=nonlin_type, norm_type=norm_type,
                                                  coord_conv=coord_conv)
        else:
            # Setup encoder
            self.encoder = Encoder(img_type=enc_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                   wide_model=wide_model, use_state=enc_inp_state, num_state=state_dim,
                                   coord_conv=coord_conv, conv_encode=False, deterministic=True)

            # Setup decoder
            self.decoder = Decoder(img_type=dec_img_type, norm_type=norm_type, nonlin_type=nonlin_type,
                                   wide_model=wide_model, pred_state=dec_pred_state, num_state=state_dim,
                                   coord_conv=coord_conv, conv_decode=False, rgb_normalize=dec_pred_norm_rgb)

            # Setup transition model
            if trans_model_type == 'nonlin':
                print("[Transition] Using non-linear transition model")
                self.transition = NonLinearTransitionModel(state_dim=self.encoder.outdim, ctrl_dim=ctrl_dim,
                                                           setting=trans_setting, predict_deltas=trans_pred_deltas,
                                                           wide_model=wide_model, nonlin_type=nonlin_type,
                                                           conv_net=False, norm_type=norm_type,
                                                           coord_conv=coord_conv)
            else:
                assert False, "Unknown transition model type: {}".format(trans_model_type)

    # Inputs are (B x (S+1) x C x H x W), (B x (S+1) x NDIM), (B x S x NDIM)
    # Outputs are lists of length (S+1) or (S) with dimensions being (B x NDIM) or (B x C x H x W)
    def forward(self, imgs, states, ctrls):
        # Encode the images and states through the encoder
        encstates = []
        for k in range(imgs.size(1)): # imgs is B x (S+1) x C x H x W
            encstate = self.encoder.forward(imgs[:,k], states[:,k]) # Predict the hidden state distribution
            encstates.append(encstate)

        # Predict a sequence of next states using the transition model
        transstates = []
        for k in range(ctrls.size(1)): # ctrls is B x S x NCTRL
            # Get the inputs
            transinpstate = encstates[0] if (k == 0) else transstates[-1] # @t=0 use encoder input, else use own prediction

            # Make a prediction through the transition model
            transoutstate = self.transition.forward(ctrls[:,k], transinpstate, None)
            transstates.append(transoutstate)

        # Decode the frames based on the encoder and transition model
        # Encoder state is used to generate frame @ t = 0, transition model states are used for t = 1 to N
        # todo: do we need to reconstruct from all encoded state samples too?
        decimgs = []
        for k in range(imgs.size(1)):
            decinp = encstates[0] if (k == 0) else transstates[k-1] # @ t = 0 reconstruct from encoder, rest from transition
            decimg = self.decoder.forward(decinp)
            decimgs.append(decimg)

        # Return stuff
        return encstates, transstates, decimgs