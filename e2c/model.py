# Global stuff
import numpy as np
import sys
import os

# Torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
add_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if add_path not in sys.path:
    sys.path.insert(0, add_path)
import ctrlnets

## todo: concat rgb/depth to pass through one conv layer, pass depth/rgb through separate conv layers & concat features

##### Create Encoder
class Encoder(nn.Module):
    def __init__(self, img_type='rgbd', norm_type='bn', nonlinearity='prelu', wide=False,
                 use_state=False, num_state=7, coord_conv=False,
                 conv_encode=False):
        super(Encoder, self).__init__()

        # Normalization
        assert norm_type == 'bn', "Unknown normalization type input: {}".format(norm_type)
        use_bn = (norm_type == 'bn')

        # Coordinate convolution
        if coord_conv:
            print('Using co-ordinate convolutional layers')
        ConvType = lambda x, y, **v: ctrlnets.BasicConv2D(in_channels=x, out_channels=y,
                                                          coord_conv=coord_conv, **v)

        # Input type
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
        chn = [input_channels, 32, 64, 128, 256, 256] if wide else [input_channels, 16, 32, 64, 128, 128]  # Num channels
        kern = [9,7,5,3,3] # Kernel sizes
        self.imgencoder = nn.Sequential(
            *[ConvType(chn[k], chn[k+1], kernel_size=kern[k], stride=1, padding=kern[k]//2,
                      use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity) for k in range(len(chn)-1)]
        )

        ###### Encode state information (jt angles, gripper position)
        self.use_state = use_state
        if self.use_state:
            sdim = [num_state, 32, 64, 64]
            self.stateencoder = nn.Sequential(
                nn.Linear(sdim[0], sdim[1]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(sdim[1], sdim[2]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(sdim[2], sdim[3]),
            )
        else:
            sdim = [0]

        ###### If we are using conv output (fully-conv basically), we can do 1x1 conv stuff
        self.conv_encode = conv_encode
        if self.conv_encode:
            # 1x1, 7x10 -> 3x5, 256
            # 1x1, 3x5 -> 3x5, 128
            # 1x1, 3x5 -> 3x5, 64
            # 1x1, 3x5 -> 3x5, 32
            print("Using convolutional output encoder with 1x1 convolutions")
            chn_out = [sdim[-1]+chn[-1], 256, 128, 64, 32] if wide else [sdim[-1]+chn[-1], 128, 64, 32, 16]
            pool = [True, False, False, False] # Pooling only for first layer to bring to 3x5
            nonlin = [nonlinearity, nonlinearity, nonlinearity, 'none'] # No non linearity for last layer
            bn = [use_bn, use_bn, use_bn, False] # No batch norm for last layer
            self.outencoder = nn.Sequential(
                *[ConvType(chn_out[k], chn_out[k+1], kernel_size=1, stride=1, padding=0,
                           use_pool=pool[k], use_bn=bn[k], nonlinearity=nonlin[k]) for k in range(len(chn_out)-1)],
            )
            self.outdim = (chn_out[-1]//2, 3, 5)
        else:
            # Fully connected output network
            print("Using fully-connected output encoder")
            odim = [sdim[-1]+(chn[-1]*7*10), 512, 512, 512] if wide else [sdim[-1]+(chn[-1]*7*10), 256, 256, 256]
            self.outencoder = nn.Sequential(
                nn.Linear(odim[0], odim[1]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(odim[1], odim[2]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(odim[2], odim[3]),
            )
            self.outdim = (odim[3]//2)

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
                h_state = h_state.view(bsz, ndim, 1, 1).expand_as(bsz, ndim, ht, wd)

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

        # Get mean & log(std) output
        mean, logstd = h_out.split(h_out.size(1)//2, dim=1) # Split into 2 along channels dim (or hidden state dim)

        # Create the co-variance matrix (as a diagonal matrix with predicted stds along the diagonal)
        stddev = torch.exp(logstd).view(bsz,-1)
        covar  = torch.stack([torch.diag(stddev[k]) for k in range(bsz)], 0) # B x N x N matrix

        # Save predictions internally
        self.predmean, self.predlogstd, self.predstd = mean, logstd, stddev

        # Return a Multi-variate normal distribution
        return torch.distributions.MultivariateNormal(loc=mean.view(bsz,-1), covariance_matrix=covar)


##### Create Decoder
class Decoder(nn.Module):
    def __init__(self, img_type='rgbd', norm_type='bn', nonlinearity='prelu', wide=False,
                 pred_state=False, num_state=7, coord_conv=False,
                 conv_decode=False, rgb_normalize=False):
        super(Decoder, self).__init__()

        # Normalization
        assert (norm_type == 'bn') or (norm_type == 'none'), "Unknown normalization type input: {}".format(norm_type)
        use_bn = (norm_type == 'bn')

        # Coordinate convolution
        if coord_conv:
            print('Using co-ordinate convolutional layers')
        ConvType   = lambda x, y, **v: ctrlnets.BasicConv2D(in_channels=x, out_channels=y,
                                                            coord_conv=coord_conv, **v)
        DeconvType = lambda x, y, **v: ctrlnets.BasicDeconv2D(in_channels=x, out_channels=y,
                                                              coord_conv=coord_conv, **v)

        # Output type
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

        ###### Decode XYZ-RGB images (half-size since we are using half size of inputs compared to before)
        # Create conv-decoder (large net => 5 conv layers with pooling)
        # 3x4, 7x10 -> 15x20
        # 4x4, 15x20 -> 30x40
        # 6x6, 30x40 -> 60x80
        # 6x6, 60x80 -> 120x160
        # 8x8, 120x160 -> 240x320
        chn = [128, 128, 64, 32, 32, output_channels] if wide else [64, 64, 32, 16, 16, output_channels]  # Num channels
        kern = [(3,4), 4, 6, 6, 8]  # Kernel sizes
        padd = [(0,1), 1, 2, 2, 3]  # Padding
        bn   = [use_bn, use_bn, use_bn, use_bn, False] # No batch norm for last layer (output)
        nonlin = [nonlinearity, nonlinearity, nonlinearity, nonlinearity, 'none'] # No non-linearity last layer (output)
        self.imgdecoder = nn.Sequential(
            *[DeconvType(chn[k], chn[k+1], kernel_size=kern[k], stride=2, padding=padd[k],
                         use_bn= bn[k], nonlinearity=nonlin[k])
              for k in range(len(chn)-1)]
        )

        # ###### Decode state information (jt angles, gripper position)
        ## todo: Add decoding of state
        # self.pred_state = pred_state
        # if self.pred_state:
        #     sdim = [self.state_chn*self.indim[1]*self.indim[2], 64, 32, num_state]
        #     self.statedecoder = nn.Sequential(
        #         nn.Linear(sdim[0], sdim[1]),
        #         ctrlnets.get_nonlinearity(nonlinearity),
        #         nn.Linear(sdim[1], sdim[2]),
        #         ctrlnets.get_nonlinearity(nonlinearity),
        #         nn.Linear(sdim[2], sdim[3]),
        #     )
        # else:
        #     sdim = [0]
        # self.state_chn = 4 if pred_state else 0 # Add a few extra channels if we are also predicting state
        assert(not self.pred_state)

        ###### If we are using conv output (fully-conv basically), we can do 1x1 conv stuff
        self.conv_decode = conv_decode
        if self.conv_decode:
            # 1x1, 3x5 -> 3x5, 16 -> 32
            # 1x1, 3x5 -> 3x5, 32 -> 64
            # 1x1, 3x5 -> 3x5, 64 -> 128
            # 1x1, 3x5 -> 7x10, 128 -> 128 + 4 (4 extra channels are inputs to the joint encoder)
            print("Using convolutional decoder with 1x1 convolutions")
            chn_in = [16, 32, 64, 128, chn[0]] if wide else [8, 16, 32, 64, chn[0]]
            self.hsdecoder = nn.Sequential(
                *[ConvType(chn_in[k], chn_in[k+1], kernel_size=1, stride=1, padding=0,
                           use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity) for k in range(len(chn_in)-2)], # Leave out last layer
                nn.UpsamplingBilinear2d((7, 10)), # Upsample to (7,10)
                ConvType(chn_in[3], chn_in[4], kernel_size=1, stride=1, padding=0,
                         use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity), # 128 x 7 x 10
            )
            self.hsdim = (chn_in[0], 3, 5)
        else:
            # Fully connected output network
            print("Using fully-connected output encoder")
            odim = [256, 256, 256, chn[0]*7*10] if wide else [128, 128, 128, chn[0]*7*10]
            self.hsdecoder = nn.Sequential(
                nn.Linear(odim[0], odim[1]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(odim[1], odim[2]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(odim[2], odim[3]),
            )
            self.hsdim = (odim[0])

    def forward(self, hidden_state):
        # Reshape hidden state to right shape
        bsz = hidden_state.size(0)
        h_state = hidden_state.view(bsz, *self.hsdim)

        # Pass it through the indecoder
        h_img = self.hsdecoder(h_state)

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

##### Create Transition Model
# todo: convolutional transition model
class NonLinearTransitionModel(nn.Module):
    def __init__(self, state_dim, ctrl_dim, norm_type='none', nonlinearity='prelu',
                 wide=False, conv_net=False):
        super(LocallyLinearTransitionModel, self).__init__()

        # Normalization
        assert (norm_type == 'bn') or (norm_type == 'none'), "Unknown normalization type input: {}".format(norm_type)
        use_bn = (norm_type == 'bn')

        #


