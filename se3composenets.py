# Global imports
import torch
import torch.nn as nn

# Local imports
import se3
import ctrlnets

####################################
### Pose-Mask Encoder (single encoder that predicts both poses and masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks and "k" poses represented as [R|t]
# NOTE: We can set a conditional flag that makes it predict both poses/masks or just poses
class PoseMaskEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', input_channels=3, num_state=7,
                 use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 wide=False, use_jt_angles=False, noise_stop_iter=1e6):
        super(PoseMaskEncoder, self).__init__()

        ###### Choose type of convolution
        ConvType   = ctrlnets.BasicConv2D
        DeconvType = ctrlnets.BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [32, 64, 128, 256, 256, 256] if wide else [8, 16, 32, 64, 128, 128]  # Num channels
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 240x320 -> 120x160
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv5 = ConvType(chn[3], chn[4], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        self.celem = chn[4] * 7 * 10

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.conv1x1 = ConvType(chn[4], chn[4], kernel_size=1, stride=1, padding=0,
                                use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
        self.deconv1 = DeconvType(chn[4], chn[3], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = DeconvType(chn[3], chn[2], kernel_size=4, stride=2, padding=1,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = DeconvType(chn[2], chn[1], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        self.deconv4 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        self.deconv5 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=8, stride=2,
                                          padding=3)  # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (wt-sharpening vs sigmoid-mask vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate # Rate for sharpening
            self.maskdecoder = ctrlnets.sharpen_masks # Use the weight-sharpener
            self.noise_stop_iter = noise_stop_iter
        else:
            self.maskdecoder = nn.Softmax2d() # SoftMax normalization

        ###### Encode jt angles
        self.use_jt_angles = use_jt_angles
        jdim = 0
        if self.use_jt_angles:
            jdim = 256
            self.jtencoder = nn.Sequential(
                nn.Linear(num_state, 128),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(128, 256),
                ctrlnets.get_nonlinearity(nonlinearity),
            )

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=False)
        self.se3_type = se3_type
        self.num_se3  = num_se3
        sdim = 256 if wide else 128 # NOTE: This was 64 before! #128
        self.se3decoder  = nn.Sequential(
                                nn.Linear(self.celem + jdim, sdim),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(sdim, self.num_se3 * self.se3_dim)  # Predict the SE3s from the conv-output
                           )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the pose-mask model to predict identity transform")
            layer = self.se3decoder[2]  # Get final SE3 prediction module
            ctrlnets.init_se3layer_identity(layer, num_se3, se3_type)  # Init to identity

    def compute_wt_sharpening_stats(self, train_iter=0):
        citer = 1 + (train_iter - self.sharpen_start_iter)
        noise_std, pow = 0, 1
        if (citer > 0):
            noise_std = min((citer/125000.0) * self.sharpen_rate, 0.1) # Should be 0.1 by ~12500 iters from start (if rate=1)
            if hasattr(self, 'noise_stop_iter') and train_iter > self.noise_stop_iter:
                noise_std = 0
            pow = min(1 + (citer/500.0) * self.sharpen_rate, 100) # Should be 26 by ~12500 iters from start (if rate=1)
        return noise_std, pow

    def forward(self, z, predict_masks=True, train_iter=0):
        if self.use_jt_angles or type(z) == list:
            x,j = z # Pts, Jt angles
        else:
            x = z

        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Run jt-encoder & concatenate the embeddings
        if self.use_jt_angles:
            j = self.jtencoder(j)
            p = torch.cat([c5.view(-1, self.celem), j], 1)
        else:
            p = c5.view(-1, self.celem)

        # Run pose-decoder to predict poses
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)

        # Run mask-decoder to predict a smooth mask
        # NOTE: Conditional based on input flag
        if predict_masks:
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

            # Return both
            return p, m
        else:
            return p

####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3quat',
                 nonlinearity='prelu', init_se3_iden=False,
                 quat_normalize=False):
        super(TransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=False) # Only if we are predicting directly
        self.num_se3 = num_se3

        # Pose encoder
        pdim = [128, 256]
        self.poseencoder = nn.Sequential(
                                nn.Linear(self.num_se3 * self.se3_dim, pdim[0]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(pdim[0], pdim[1]),
                                ctrlnets.get_nonlinearity(nonlinearity)
                            )

        # Control encoder
        cdim = [128, 256] #[64, 128]
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, cdim[0]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(cdim[0], cdim[1]),
                                ctrlnets.get_nonlinearity(nonlinearity)
                            )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(pdim[1]+cdim[1], 256),
            ctrlnets.get_nonlinearity(nonlinearity),
            nn.Linear(256, 128),
            ctrlnets.get_nonlinearity(nonlinearity),
            nn.Linear(128, self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            layer = self.deltase3decoder[4]  # Get final SE3 prediction module
            ctrlnets.init_se3layer_identity(layer, num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.se3_type    = se3_type

        # Compose deltas with prev pose to get next pose
        # It predicts the delta transform of the object between time "t1" and "t2": p_t2_to_t1: takes a point in t2 and converts it to t1's frame of reference
	    # So, the full transform is: p_t2 = p_t1 * p_t2_to_t1 (or: p_t2 (t2 to cam) = p_t1 (t1 to cam) * p_t2_to_t1 (t2 to t1))
        if self.se3_type == 'se3quat':
            self.posedecoder = lambda x, y: se3.ComposeSE3QuatPair(x, y, normalize=quat_normalize)
        else:
            assert False, 'Unknown SE3 type input: {}'.format(self.se3_type)

    def forward(self, x):
        # Encode inputs
        p, c = x # Pose, Control
        pv = p.view(-1, self.num_se3 * self.se3_dim) # Reshape pose
        pe = self.poseencoder(pv)    # Encode pose
        ce = self.ctrlencoder(c)     # Encode ctrl
        x = torch.cat([pe,ce], 1)    # Concatenate encoded vectors

        # Predict delta SE3
        x = self.deltase3decoder(x)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)

        # Predicted delta is already in the global frame of reference, use it directly (from global to global)
        z = self.posedecoder(x, p) # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)
        y = x # D = SE3_2 * SE3_1^-1 (global to global)

        # Return
        return [y, z] # Return both the deltas (in global frame) and the composed next pose

####################################
### SE3-Pose-Nets that composes directly in the SE3 space, not converting to R|t
### Currently, this has a separate pose & mask predictor as well as a transition model
### NOTE: The forward pass is not currently implemented, this needs to be done outside
class SE3PoseComposeModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3quat', input_channels=3,
                 use_bn=True, nonlinearity='prelu', num_state=7,
                 init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 wide=False, use_jt_angles=False, noise_stop_iter=1e6,
                 quat_normalize=False):
        super(SE3PoseComposeModel, self).__init__()

        # Initialize the pose & mask model
        self.posemaskmodel = PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, num_state=num_state,
                                             input_channels=input_channels, init_se3_iden=init_posese3_iden,
                                             use_bn=use_bn, nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                             sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                             wide=wide, use_jt_angles=use_jt_angles, noise_stop_iter=noise_stop_iter)

        # Initialize the transition model
        self.transitionmodel = TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, se3_type=se3_type,
                                               nonlinearity=nonlinearity, init_se3_iden=init_transse3_iden,
                                               quat_normalize=quat_normalize)

        # Options
        self.use_jt_angles = use_jt_angles

    # Predict pose only
    def forward_only_pose(self, x):
        ptcloud, jtangles = x
        inp = [ptcloud, jtangles] if self.use_jt_angles else ptcloud
        return self.posemaskmodel(inp, predict_masks=False) # returns only pose

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        ptcloud, jtangles = x
        inp = [ptcloud, jtangles] if self.use_jt_angles else ptcloud
        return self.posemaskmodel(inp, train_iter=train_iter, predict_masks=True) # Predict both

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl):
        return self.transitionmodel([pose, ctrl])

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for SE3PoseComposeModel is not yet implemented')
        raise NotImplementedError
