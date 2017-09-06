import torch
import torch.nn as nn
import se3layers as se3nn
import ctrlnets

################################################################################
'''
    Single step / Recurrent models
'''

####################################
### Pose Encoder
# Model that takes in "depth/point cloud" to generate "k" poses represented as [R|t]
class SE2PoseEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, use_kinchain=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 wide=False, use_jt_angles=False, num_state=7):
        super(SE2PoseEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('[PoseEncoder] Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = ctrlnets.PreConv2D if pre_conv else ctrlnets.BasicConv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [16, 32, 64, 64]
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        self.celem = chn[3] * 7 * 10

        ###### Encode jt angles
        self.use_jt_angles = use_jt_angles
        jdim = 0
        if use_jt_angles:
            jdim = 256
            self.jtencoder = nn.Sequential(
                nn.Linear(num_state, 128),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(128, 256),
                ctrlnets.get_nonlinearity(nonlinearity),
            )

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        sdim = 128
        self.se3decoder  = nn.Sequential(
                                nn.Linear(self.celem + jdim, sdim),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(sdim, self.num_se3 * self.se3_dim) # Predict the SE3s from the conv-output
                           )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the pose-mask model to predict identity transform")
            layer = self.se3decoder[2]  # Get final SE3 prediction module
            ctrlnets.init_se3layer_identity(layer, num_se3, se3_type)  # Init to identity

        # Create pose decoder (convert to r/t)
        self.posedecoder = nn.Sequential()
        self.posedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, use_pivot)) # Convert to Rt
        if use_pivot:
            self.posedecoder.add_module('pivotrt', se3nn.CollapseRtPivots()) # Collapse pivots
        if use_kinchain:
            self.posedecoder.add_module('kinchain', se3nn.ComposeRt(rightToLeft=False)) # Kinematic chain

    def forward(self, z):
        if self.use_jt_angles:
            x,j = z # Pts, Jt angles
        else:
            x = z

        # Run conv-encoder to generate embedding
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Run jt-encoder & concatenate the embeddings
        if self.use_jt_angles:
            j = self.jtencoder(j)
            p = torch.cat([x.view(-1, self.celem), j], 1)
        else:
            p = x.view(-1, self.celem)

        # Run pose-decoder to predict poses
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Return poses
        return p

####################################
### Mask Encoder (single encoder that takes a depth image and predicts segmentation masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks
class SE2MaskEncoder(nn.Module):
    def __init__(self, num_se3, pre_conv=False, input_channels=3, use_bn=True, nonlinearity='prelu',
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, wide=False):
        super(SE2MaskEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType = ctrlnets.PreConv2D if pre_conv else ctrlnets.BasicConv2D
        DeconvType = ctrlnets.PreDeconv2D if pre_conv else ctrlnets.BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [16, 32, 64, 64]
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        self.celem = chn[3] * 7 * 10

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.conv1x1 = ConvType(chn[3], chn[3], kernel_size=1, stride=1, padding=0,
                                use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
        self.deconv1 = DeconvType(chn[3], chn[2], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = DeconvType(chn[2], chn[1], kernel_size=4, stride=2, padding=1,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        if pre_conv:
            self.deconv4 = DeconvType(chn[0], num_se3, kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        else:
            self.deconv4 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=6, stride=2,
                                              padding=2)  # 6x6, 60x80 -> 120x160

        # Normalize to generate mask (wt-sharpening vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter  # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate  # Rate for sharpening
            self.maskdecoder = ctrlnets.sharpen_masks  # Use the weight-sharpener
        elif use_sigmoid_mask:
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

        # Run mask-decoder to predict a smooth mask
        m = self.conv1x1(c4)
        m = self.deconv1(m, c3)
        m = self.deconv2(m, c2)
        m = self.deconv3(m, c1)
        m = self.deconv4(m)

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
class SE2PoseMaskEncoder(nn.Module):
    def __init__(self, num_se3, se3_type='se3aa', use_pivot=False, use_kinchain=False, pre_conv=False,
                 input_channels=3, use_bn=True, nonlinearity='prelu', init_se3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, wide=False, use_jt_angles=False, num_state=7):
        super(SE2PoseMaskEncoder, self).__init__()

        ###### Choose type of convolution
        if pre_conv:
            print('Using BN + Non-Linearity + Conv/Deconv architecture')
        ConvType   = ctrlnets.PreConv2D if pre_conv else ctrlnets.BasicConv2D
        DeconvType = ctrlnets.PreDeconv2D if pre_conv else ctrlnets.BasicDeconv2D

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [16, 32, 64, 64]
        self.conv1 = ConvType(input_channels, chn[0], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 120x160 -> 60x80
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 60x80 -> 30x40
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 30x40 -> 15x20
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 15x20 -> 7x10
        self.celem = chn[3] * 7 * 10

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

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.conv1x1 = ConvType(chn[3], chn[3], kernel_size=1, stride=1, padding=0,
                                use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 1x1, 7x10 -> 7x10
        self.deconv1 = DeconvType(chn[3], chn[2], kernel_size=(3, 4), stride=2, padding=(0, 1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 3x4, 7x10 -> 15x20
        self.deconv2 = DeconvType(chn[2], chn[1], kernel_size=4, stride=2, padding=1,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 4x4, 15x20 -> 30x40
        self.deconv3 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 30x40 -> 60x80
        if pre_conv:
            self.deconv4 = DeconvType(chn[0], num_se3, kernel_size=6, stride=2, padding=2,
                                      use_bn=use_bn, nonlinearity=nonlinearity)  # 6x6, 60x80 -> 120x160
        else:
            self.deconv4 = nn.ConvTranspose2d(chn[0], num_se3, kernel_size=6, stride=2, padding=2)  # 6x6, 60x80 -> 120x160

        # Normalize to generate mask (wt-sharpening vs sigmoid-mask vs soft-mask model)
        self.use_wt_sharpening = use_wt_sharpening
        if use_wt_sharpening:
            self.sharpen_start_iter = sharpen_start_iter # Start iter for the sharpening
            self.sharpen_rate = sharpen_rate # Rate for sharpening
            self.maskdecoder = ctrlnets.sharpen_masks # Use the weight-sharpener
        elif use_sigmoid_mask:
            self.maskdecoder = nn.Sigmoid() # No normalization, each pixel can belong to multiple masks but values between (0-1)
        else:
            self.maskdecoder = nn.Softmax2d() # SoftMax normalization

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3
        sdim = 128 # NOTE: This was 64 before! #128
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

    def forward(self, z, predict_masks=True, train_iter=0):
        if self.use_jt_angles:
            x,j = z # Pts, Jt angles
        else:
            x = z

        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        # Run jt-encoder & concatenate the embeddings
        if self.use_jt_angles:
            j = self.jtencoder(j)
            p = torch.cat([c4.view(-1, self.celem), j], 1)
        else:
            p = c4.view(-1, self.celem)

        # Run pose-decoder to predict poses
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Run mask-decoder to predict a smooth mask
        # NOTE: Conditional based on input flag
        if predict_masks:
            m = self.conv1x1(c4)
            m = self.deconv1(m, c3)
            m = self.deconv2(m, c2)
            m = self.deconv3(m, c1)
            m = self.deconv4(m)

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
class SE2TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, use_pivot=False, se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False, use_jt_angles=False, num_state=7):
        super(SE2TransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=use_pivot)
        self.num_se3 = num_se3

        # Pose encoder
        pdim = [64, 128]
        self.poseencoder = nn.Sequential(
                                nn.Linear(self.num_se3 * 12, pdim[0]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(pdim[0], pdim[1]),
                                ctrlnets.get_nonlinearity(nonlinearity)
                            )

        # Control encoder
        cdim = [64, 128] #[64, 128]
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, cdim[0]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(cdim[0], cdim[1]),
                                ctrlnets.get_nonlinearity(nonlinearity)
                            )

        # Jt angle encoder
        jdim = [0, 0]
        self.use_jt_angles = use_jt_angles
        if use_jt_angles:
            jdim = [128, 256]  # [64, 128]
            self.jtangleencoder = nn.Sequential(
                nn.Linear(num_state, jdim[0]),
                ctrlnets.get_nonlinearity(nonlinearity),
                nn.Linear(jdim[0], jdim[1]),
                ctrlnets.get_nonlinearity(nonlinearity)
            )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(pdim[1]+cdim[1]+jdim[1], 256),
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
        if self.use_jt_angles:
            p, j, c = x  # Pose, Jtangles, Control
        else:
            p, c = x  # Pose, Control
        pv = p.view(-1, self.num_se3*12) # Reshape pose
        pe = self.poseencoder(pv)    # Encode pose
        ce = self.ctrlencoder(c)     # Encode ctrl
        if self.use_jt_angles:
            je = self.jtangleencoder(j)  # Encode jt angles
            x = torch.cat([pe,je,ce], 1) # Concatenate encoded vectors
        else:
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
class SE2PoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(SE2PoseModel, self).__init__()

        # Initialize the pose-mask model
        self.posemaskmodel = SE2PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                             use_kinchain=use_kinchain, input_channels=input_channels,
                                             init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                             nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                             sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                             use_sigmoid_mask=use_sigmoid_mask, wide=wide,
                                             use_jt_angles=use_jt_angles, num_state=num_state)
        # Initialize the transition model
        self.transitionmodel = SE2TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden,
                                               local_delta_se3=local_delta_se3,
                                               use_jt_angles=use_jt_angles_trans, num_state=num_state)

        # Options
        self.use_jt_angles = use_jt_angles
        self.use_jt_angles_trans = use_jt_angles_trans

    # Forward pass through the model
    def forward(self, x, train_iter=0):
        # Get input vars
        ptcloud_1, ptcloud_2, ctrl_1, jtangles_1, jtangles_2 = x

        # Get pose & mask predictions @ t0 & t1
        inp1 = [ptcloud_1, jtangles_1] if self.use_jt_angles else ptcloud_1
        inp2 = [ptcloud_2, jtangles_2] if self.use_jt_angles else ptcloud_2
        pose_1, mask_1 = self.posemaskmodel(inp1, train_iter=train_iter, predict_masks=True)  # ptcloud @ t1
        pose_2, mask_2 = self.posemaskmodel(inp2, train_iter=train_iter, predict_masks=True)  # ptcloud @ t2

        # Get transition model predicton of pose_1
        inp3 = [pose_1, jtangles_1, ctrl_1] if self.use_jt_angles_trans else [pose_1, ctrl_1]
        deltapose_t_12, pose_t_2 = self.transitionmodel(inp3)  # Predicts [delta-pose, pose]

        # Return outputs
        return [pose_1, mask_1], [pose_2, mask_2], [deltapose_t_12, pose_t_2]

####################################
### SE3-OnlyPose-Model (single-step model that takes [depth_t, depth_t+1, ctrl-t] to predict
### pose_t, pose_t+1, [delta-pose, poset_t+1]
class SE2OnlyPoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_posese3_iden=False, init_transse3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(SE2OnlyPoseModel, self).__init__()

        # Initialize the pose-mask model
        self.posemodel = SE2PoseEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                     use_kinchain=use_kinchain, input_channels=input_channels,
                                     init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                     nonlinearity=nonlinearity, wide=wide,
                                     use_jt_angles=use_jt_angles, num_state=num_state)
        # Initialize the transition model
        self.transitionmodel = SE2TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden=init_transse3_iden,
                                               local_delta_se3=local_delta_se3,
                                               use_jt_angles=use_jt_angles_trans, num_state=num_state)

        # Options
        self.use_jt_angles = use_jt_angles
        self.use_jt_angles_trans = use_jt_angles_trans

    # Forward pass through the model
    def forward(self, x):
        # Get input vars
        ptcloud_1, ptcloud_2, ctrl_1, jtangles_1, jtangles_2 = x

        # Get pose & mask predictions @ t0 & t1
        inp1 = [ptcloud_1, jtangles_1] if self.use_jt_angles else ptcloud_1
        inp2 = [ptcloud_2, jtangles_2] if self.use_jt_angles else ptcloud_2
        pose_1 = self.posemodel(inp1)  # ptcloud @ t1
        pose_2 = self.posemodel(inp2)  # ptcloud @ t2

        # Get transition model predicton of pose_1
        inp3 = [pose_1, jtangles_1, ctrl_1] if self.use_jt_angles_trans else [pose_1, ctrl_1]
        deltapose_t_12, pose_t_2 = self.transitionmodel(inp3)  # Predicts [delta-pose, pose]

        # Return outputs
        return pose_1, pose_2, [deltapose_t_12, pose_t_2]

####################################
### SE3-OnlyMask-Model (single-step model that takes [depth_t, depth_t+1, ctrl-t] to predict
### mask_t, mask_t+1
class SE2OnlyMaskModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_posese3_iden=False, init_transse3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(SE2OnlyMaskModel, self).__init__()

        # Initialize the pose-mask model
        self.maskmodel = SE2MaskEncoder(num_se3=num_se3, input_channels=input_channels,
                                     use_bn=use_bn, pre_conv=pre_conv,
                                     nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                     sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                     use_sigmoid_mask=use_sigmoid_mask, wide=wide)

    # Forward pass through the model
    def forward(self, x, train_iter=0):
        # Get input vars
        ptcloud_1, ptcloud_2 = x

        # Get pose & mask predictions @ t0 & t1
        mask_1 = self.maskmodel(ptcloud_1, train_iter=train_iter)  # ptcloud @ t1
        mask_2 = self.maskmodel(ptcloud_2, train_iter=train_iter)  # ptcloud @ t2

        # Return outputs
        return mask_1, mask_2

####################################
### SE3-Pose-Model (single-step model that takes [depth_t, depth_t+1, ctrl-t] to predict
### [pose_t, mask_t], [pose_t+1, mask_t+1], [delta-pose, poset_t+1]
class SE2DecompModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False,
                 nonlinearity='prelu', init_posese3_iden=False, init_transse3_iden=False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(SE2DecompModel, self).__init__()

        # Initialize the pose model
        self.posemodel = SE2PoseEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                     use_kinchain=use_kinchain, input_channels=input_channels,
                                     init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                     nonlinearity=nonlinearity, wide=wide,
                                     use_jt_angles=use_jt_angles, num_state=num_state)

        # Initialize the mask model
        self.maskmodel = SE2MaskEncoder(num_se3=num_se3, input_channels=input_channels,
                                     use_bn=use_bn, pre_conv=pre_conv,
                                     nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                     sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                     use_sigmoid_mask=use_sigmoid_mask, wide=wide)

        # Initialize the transition model
        self.transitionmodel = SE2TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden=init_transse3_iden,
                                               local_delta_se3=local_delta_se3,
                                               use_jt_angles=use_jt_angles_trans, num_state=num_state)

        # Options
        self.use_jt_angles = use_jt_angles
        self.use_jt_angles_trans = use_jt_angles_trans

    # Forward pass through the model
    def forward(self, x, train_iter=0):
        # Get input vars
        ptcloud_1, ptcloud_2, ctrl_1, jtangles_1, jtangles_2 = x

        # Get pose predictions @ t0 & t1
        inp1 = [ptcloud_1, jtangles_1] if self.use_jt_angles else ptcloud_1
        inp2 = [ptcloud_2, jtangles_2] if self.use_jt_angles else ptcloud_2
        pose_1 = self.posemodel(inp1)  # ptcloud @ t1
        pose_2 = self.posemodel(inp2)  # ptcloud @ t2

        # Get mask predictions @ t0 & t1
        mask_1 = self.maskmodel(ptcloud_1, train_iter=train_iter)  # ptcloud @ t1
        mask_2 = self.maskmodel(ptcloud_2, train_iter=train_iter)  # ptcloud @ t2

        # Get transition model predicton of pose_1
        inp3 = [pose_1, jtangles_1, ctrl_1] if self.use_jt_angles_trans else [pose_1, ctrl_1]
        deltapose_t_12, pose_t_2 = self.transitionmodel(inp3)  # Predicts [delta-pose, pose]

        # Return outputs
        return [pose_1, mask_1], [pose_2, mask_2], [deltapose_t_12, pose_t_2]

####################################
### Multi-step version of the SE3-Pose-Model
### Currently, this has a separate pose & mask predictor as well as a transition model
### NOTE: The forward pass is not currently implemented, this needs to be done outside
class MultiStepSE2PoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False, decomp_model=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(MultiStepSE2PoseModel, self).__init__()

        # Initialize the pose & mask model
        self.decomp_model = decomp_model
        if self.decomp_model:
            print('Using separate networks for pose and mask prediction')
            self.posemodel = SE2PoseEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                         use_kinchain=use_kinchain, input_channels=input_channels,
                                         init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity, wide=wide,
                                         use_jt_angles=use_jt_angles, num_state=num_state)
            self.maskmodel = SE2MaskEncoder(num_se3=num_se3, input_channels=input_channels,
                                         use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                         sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                         use_sigmoid_mask=use_sigmoid_mask, wide=wide)
        else:
            print('Using single network for pose & mask prediction')
            self.posemaskmodel = SE2PoseMaskEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                                 use_kinchain=use_kinchain, input_channels=input_channels,
                                                 init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                                 nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                                 sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                                 use_sigmoid_mask=use_sigmoid_mask, wide=wide,
                                                 use_jt_angles=use_jt_angles, num_state=num_state)

        # Initialize the transition model
        self.transitionmodel = SE2TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                               se3_type=se3_type, use_kinchain=use_kinchain,
                                               nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden,
                                               local_delta_se3=local_delta_se3,
                                               use_jt_angles=use_jt_angles_trans, num_state=num_state)

        # Options
        self.use_jt_angles = use_jt_angles
        self.use_jt_angles_trans = use_jt_angles_trans

    # Predict pose only
    def forward_only_pose(self, x):
        ptcloud, jtangles = x
        inp = [ptcloud, jtangles] if self.use_jt_angles else ptcloud
        if self.decomp_model:
            return self.posemodel(inp)
        else:
            return self.posemaskmodel(inp, predict_masks=False)  # returns only pose

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        ptcloud, jtangles = x
        inp = [ptcloud, jtangles] if self.use_jt_angles else ptcloud
        if self.decomp_model:
            pose = self.posemodel(inp)
            mask = self.maskmodel(ptcloud, train_iter=train_iter)
            return pose, mask
        else:
            return self.posemaskmodel(inp, train_iter=train_iter, predict_masks=True)  # Predict both

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl, jtangles=None):
        if self.use_jt_angles_trans:
            return self.transitionmodel([pose, jtangles, ctrl])
        else:
            return self.transitionmodel([pose, ctrl])

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for Multi-Step SE3-Pose-Model is not yet implemented')
        raise NotImplementedError

####################################
### Multi-step version of the SE3-OnlyPose-Model
### Predicts only poses, uses GT masks
### NOTE: The forward pass is not currently implemented, this needs to be done outside
class MultiStepSE2OnlyPoseModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False, decomp_model=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(MultiStepSE2OnlyPoseModel, self).__init__()

        # Initialize the pose & mask model
        self.posemodel = SE2PoseEncoder(num_se3=num_se3, se3_type=se3_type, use_pivot=use_pivot,
                                        use_kinchain=use_kinchain, input_channels=input_channels,
                                        init_se3_iden=init_posese3_iden, use_bn=use_bn, pre_conv=pre_conv,
                                        nonlinearity=nonlinearity, wide=wide,
                                        use_jt_angles=use_jt_angles, num_state=num_state)


        # Initialize the transition model
        self.transitionmodel = SE2TransitionModel(num_ctrl=num_ctrl, num_se3=num_se3, use_pivot=use_pivot,
                                                  se3_type=se3_type, use_kinchain=use_kinchain,
                                                  nonlinearity=nonlinearity, init_se3_iden = init_transse3_iden,
                                                  local_delta_se3=local_delta_se3,
                                                  use_jt_angles=use_jt_angles_trans, num_state=num_state)
    # Predict pose only
    def forward_only_pose(self, x):
        return self.posemodel(x)

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        print('Only pose model does not predict masks')
        raise NotImplementedError

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl, jtangles=None):
        if self.use_jt_angles_trans:
            return self.transitionmodel([pose, jtangles, ctrl])
        else:
            return self.transitionmodel([pose, ctrl])

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for Multi-Step SE3-Pose-Model is not yet implemented')
        raise NotImplementedError

####################################
### Multi-step version of the SE3-OnlyMask-Model (Only predicts mask)
class MultiStepSE2OnlyMaskModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False,
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False, decomp_model=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7):
        super(MultiStepSE2OnlyMaskModel, self).__init__()

        # Initialize the mask model
        self.maskmodel = SE2MaskEncoder(num_se3=num_se3, input_channels=input_channels,
                                         use_bn=use_bn, pre_conv=pre_conv,
                                         nonlinearity=nonlinearity, use_wt_sharpening=use_wt_sharpening,
                                         sharpen_start_iter=sharpen_start_iter, sharpen_rate=sharpen_rate,
                                         use_sigmoid_mask=use_sigmoid_mask, wide=wide)

    # Predict mask only
    def forward_only_mask(self, x, train_iter=0):
        mask = self.maskmodel(x, train_iter=train_iter)
        return mask

    # Predict pose only
    def forward_only_pose(self, x):
        raise NotImplementedError

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        raise NotImplementedError

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl):
        raise NotImplementedError

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for Multi-Step SE2-Pose-Model is not yet implemented')
        raise NotImplementedError