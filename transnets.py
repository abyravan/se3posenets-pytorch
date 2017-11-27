import torch
import torch.nn as nn
import se3layers as se3nn
import ctrlnets

####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, delta_pivot='', se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False, use_jt_angles=False, num_state=7, use_bn=False):
        super(TransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=(delta_pivot == 'pred')) # Only if we are predicting directly
        self.num_se3 = num_se3

        # Pose encoder
        pdim = [128, 256]
        self.poseencoder = nn.Sequential(
            ctrlnets.BasicLinear(self.num_se3 * 12, pdim[0],
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(pdim[0], pdim[1],
                                 use_bn=use_bn, nonlinearity=nonlinearity),
                            )

        # Control encoder
        cdim = [128, 256] #[64, 128]
        self.ctrlencoder = nn.Sequential(
            ctrlnets.BasicLinear(num_ctrl, cdim[0],
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(cdim[0], cdim[1],
                                 use_bn=use_bn, nonlinearity=nonlinearity),
                            )
        self.use_jt_angles=False
        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            ctrlnets.BasicLinear(pdim[1]+cdim[1], 256,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(256, 128,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            nn.Linear(128, self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            ctrlnets.init_se3layer_identity(self.deltase3decoder[-1], num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.delta_pivot = delta_pivot
        self.inp_pivot   = (self.delta_pivot != '') and (self.delta_pivot != 'pred') # Only for these 2 cases, no pivot is passed in as input
        self.deltaposedecoder = nn.Sequential()
        self.deltaposedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, (self.delta_pivot != '')))  # Convert to Rt
        if (self.delta_pivot != ''):
            self.deltaposedecoder.add_module('pivotrt', se3nn.CollapseRtPivots())  # Collapse pivots
        #if use_kinchain:
        #    self.deltaposedecoder.add_module('kinchain', se3nn.ComposeRt(rightToLeft=False))  # Kinematic chain

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
            if self.inp_pivot:
                p, j, c, pivot = x # Pose, Jtangles, Control, Pivot
            else:
                p, j, c = x # Pose, Jtangles, Control
        else:
            if self.inp_pivot:
                p, c, pivot = x # Pose, Control, Pivot
            else:
                p, c = x # Pose, Control
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
        if self.inp_pivot: # For these two cases, we don't need to handle anything
            x = torch.cat([x, pivot.view(-1, self.num_se3, 3)], 2) # Use externally provided pivots
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
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class SimpleTransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, delta_pivot='', se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False, use_jt_angles=False, num_state=7, use_bn=False, wide=False):
        super(SimpleTransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=(delta_pivot == 'pred')) # Only if we are predicting directly
        self.num_se3 = num_se3

        # Simple linear network (concat the two, run 2 layers with 1 nonlinearity)
        pdim = [1024, 512] if wide else [256, 256]
        self.deltase3decoder = nn.Sequential(
            ctrlnets.BasicLinear(self.num_se3*12 + num_ctrl, pdim[0],
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(pdim[0], pdim[1],
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            nn.Linear(pdim[1], self.num_se3*self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            ctrlnets.init_se3layer_identity(self.deltase3decoder[-1], num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.deltaposedecoder = se3nn.SE3ToRt(se3_type, (delta_pivot != ''))  # Convert to Rt

        # Compose deltas with prev pose to get next pose
        # It predicts the delta transform of the object between time "t1" and "t2": p_t2_to_t1: takes a point in t2 and converts it to t1's frame of reference
	    # So, the full transform is: p_t2 = p_t1 * p_t2_to_t1 (or: p_t2 (t2 to cam) = p_t1 (t1 to cam) * p_t2_to_t1 (t2 to t1))
        self.posedecoder = se3nn.ComposeRtPair()

    def forward(self, x):
        # Run the forward pass
        p, c = x # Pose, Control
        x = torch.cat([p.view(-1, self.num_se3*12), c], 1) # Concatenate inputs
        x = self.deltase3decoder(x)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)
        x = self.deltaposedecoder(x) # Convert delta-SE3 to delta-Pose (can be in local or global frame of reference)

        # Predicted delta is already in the global frame of reference, use it directly (from global to global)
        z = self.posedecoder(x, p) # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)
        y = x # D = SE3_2 * SE3_1^-1 (global to global)

        # Return
        return [y, z] # Return both the deltas (in global frame) and the composed next pose

####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class SimpleDenseNetTransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, delta_pivot='', se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False, use_jt_angles=False, num_state=7, use_bn=False):
        super(SimpleDenseNetTransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=(delta_pivot == 'pred'))  # Only if we are predicting directly
        self.num_se3 = num_se3

        # Simple linear network (concat the two, run 2 layers with 1 nonlinearity)
        idim = (self.num_se3 * 12) + num_ctrl
        odim = (self.num_se3 * self.se3_dim)
        pdim = [128, 128, 128]
        self.l0 = ctrlnets.BasicLinear(idim,                  pdim[0],
                                 use_bn=use_bn, nonlinearity=nonlinearity)
        self.l1 = ctrlnets.BasicLinear(idim+pdim[0],          pdim[1],
                                 use_bn=use_bn, nonlinearity=nonlinearity)
        self.l2 = ctrlnets.BasicLinear(idim+pdim[0]+pdim[1],  pdim[2],
                                 use_bn=use_bn, nonlinearity=nonlinearity)
        self.l3 = nn.Linear(idim+sum(pdim), odim)

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            ctrlnets.init_se3layer_identity(self.l3, num_se3, se3_type)  # Init to identity

        # Create pose decoder (convert to r/t)
        self.deltaposedecoder = se3nn.SE3ToRt(se3_type, (delta_pivot != ''))  # Convert to Rt

        # Compose deltas with prev pose to get next pose
        # It predicts the delta transform of the object between time "t1" and "t2": p_t2_to_t1: takes a point in t2 and converts it to t1's frame of reference
        # So, the full transform is: p_t2 = p_t1 * p_t2_to_t1 (or: p_t2 (t2 to cam) = p_t1 (t1 to cam) * p_t2_to_t1 (t2 to t1))
        self.posedecoder = se3nn.ComposeRtPair()

    def forward(self, x):
        # Get inputs
        p, c = x  # Pose, Control

        # Run through the dense-FC net
        i = torch.cat([p.view(-1, self.num_se3 * 12), c], 1)  # Concatenate inputs
        y0 = self.l0(i)
        z0 = torch.cat([x, y0], 1) # x (+) y0
        y1 = self.l1(z0)
        z1 = torch.cat([z0,y1], 1) # x (+) y0 (+) y1
        y2 = self.l2(z1)
        z2 = torch.cat([z1,y2], 1) # x (+) y0 (+) y1 (+) y2
        o  = self.l3(z2)           # Predict delta-SE3

        # Convert delta-SE3 to delta-Pose (can be in local or global frame of reference)
        o  = o.view(-1, self.num_se3, self.se3_dim)
        o  = self.deltaposedecoder(o)

        # Predicted delta is already in the global frame of reference, use it directly (from global to global)
        z = self.posedecoder(o,p)  # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)
        y = o  # D = SE3_2 * SE3_1^-1 (global to global)

        # Return
        return [y, z]  # Return both the deltas (in global frame) and the composed next pose

####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class DeepTransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, delta_pivot='', se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False, use_jt_angles=False, num_state=7, use_bn=False):
        super(DeepTransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=(delta_pivot == 'pred')) # Only if we are predicting directly
        self.num_se3 = num_se3

        # Pose encoder
        self.poseencoder = nn.Sequential(
            ctrlnets.BasicLinear(self.num_se3 * 12, 256,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(256, 512,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(512, 1024,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
                            )

        # Control encoder
        self.ctrlencoder = nn.Sequential(
            ctrlnets.BasicLinear(num_ctrl, 256,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(256, 512,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(512, 1024,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
        )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            ctrlnets.BasicLinear(1024+1024, 1024,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(1024, 512,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            ctrlnets.BasicLinear(512, 512,
                                 use_bn=use_bn, nonlinearity=nonlinearity),
            nn.Linear(512, self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            layer = self.deltase3decoder[-1]  # Get final SE3 prediction module
            ctrlnets.init_se3layer_identity(layer, num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.delta_pivot = delta_pivot
        self.inp_pivot   = (self.delta_pivot != '') and (self.delta_pivot != 'pred') # Only for these 2 cases, no pivot is passed in as input
        self.deltaposedecoder = nn.Sequential()
        self.deltaposedecoder.add_module('se3rt', se3nn.SE3ToRt(se3_type, (self.delta_pivot != '')))  # Convert to Rt
        if (self.delta_pivot != ''):
            self.deltaposedecoder.add_module('pivotrt', se3nn.CollapseRtPivots())  # Collapse pivots
        #if use_kinchain:
        #    self.deltaposedecoder.add_module('kinchain', se3nn.ComposeRt(rightToLeft=False))  # Kinematic chain

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
        self.use_jt_angles=False
        if self.use_jt_angles:
            if self.inp_pivot:
                p, j, c, pivot = x # Pose, Jtangles, Control, Pivot
            else:
                p, j, c = x # Pose, Jtangles, Control
        else:
            if self.inp_pivot:
                p, c, pivot = x # Pose, Control, Pivot
            else:
                p, c = x # Pose, Control
        pv = p.view(-1, self.num_se3*12) # Reshape pose
        pe = self.poseencoder(pv)    # Encode pose
        ce = self.ctrlencoder(c)     # Encode ctrl
        x = torch.cat([pe,ce], 1)    # Concatenate encoded vectors
        x = self.deltase3decoder(x)  # Predict delta-SE3
        x = x.view(-1, self.num_se3, self.se3_dim)
        if self.inp_pivot: # For these two cases, we don't need to handle anything
            x = torch.cat([x, pivot.view(-1, self.num_se3, 3)], 2) # Use externally provided pivots
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
### Multi-step version of the SE3-OnlyMask-Model (Only predicts mask)
class MultiStepSE3OnlyTransModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, se3_type='se3aa', use_pivot=False, delta_pivot='',
                 use_kinchain=False, input_channels=3, use_bn=True, pre_conv=False, decomp_model=False,
                 nonlinearity='prelu', init_posese3_iden= False, init_transse3_iden = False,
                 use_wt_sharpening=False, sharpen_start_iter=0, sharpen_rate=1,
                 use_sigmoid_mask=False, local_delta_se3=False, wide=False,
                 use_jt_angles=False, use_jt_angles_trans=False, num_state=7,
                 full_res=False, trans_type='default', trans_bn=False):
        super(MultiStepSE3OnlyTransModel, self).__init__()

        # Initialize the transition model
        bn_str = ' with batch normalization enabled' if trans_bn else ''
        if trans_type == 'default':
            print('==> Using [DEFAULT] transition model'+bn_str)
            transmodelfn = TransitionModel
        elif trans_type == 'simple':
            print('==> Using [SIMPLE] transition model'+bn_str)
            transmodelfn = lambda **v: SimpleTransitionModel(wide=False, **v)
        elif trans_type == 'simplewide':
            print('==> Using [SIMPLE-WIDE] transition model'+bn_str)
            transmodelfn = lambda **v: SimpleTransitionModel(wide=True, **v)
        elif trans_type == 'simpledense':
            print('==> Using [SIMPLE-DENSE] transition model'+bn_str)
            transmodelfn = SimpleDenseNetTransitionModel
        elif trans_type == 'deep':
            print('==> Using [DEEP] transition model'+bn_str)
            transmodelfn = DeepTransitionModel
        else:
            assert False, "Unknown transition model type: {}".format(trans_type)
        self.transitionmodel = transmodelfn(num_ctrl=num_ctrl, num_se3=num_se3, delta_pivot=delta_pivot,
                                            se3_type=se3_type, use_kinchain=use_kinchain,
                                            nonlinearity=nonlinearity, init_se3_iden=init_transse3_iden,
                                            local_delta_se3=local_delta_se3,
                                            use_jt_angles=use_jt_angles_trans, num_state=num_state,
                                            use_bn=trans_bn)

    # Predict mask only
    def forward_only_mask(self, x, train_iter=0):
        raise NotImplementedError

    # Predict pose only
    def forward_only_pose(self, x):
        raise NotImplementedError

    # Predict both pose and mask
    def forward_pose_mask(self, x, train_iter=0):
        raise NotImplementedError

    # Predict next pose based on current pose and control
    def forward_next_pose(self, pose, ctrl, jtangles=None, pivots=None):
        inp = [pose,ctrl]
        return self.transitionmodel(inp)

    # Forward pass through the model
    def forward(self, x):
        print('Forward pass for Multi-Step SE3-Pose-Model is not yet implemented')
        raise NotImplementedError
