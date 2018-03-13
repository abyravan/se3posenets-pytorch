import torch
import torch.nn as nn

import ctrlnets
import se3layers as se3nn

####################################
### Transition model (predicts change in poses based on the applied control)
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class PosVelTransitionModel(nn.Module):
    def __init__(self, num_ctrl, num_se3, delta_pivot='', se3_type='se3aa',
                 use_kinchain=False, nonlinearity='prelu', init_se3_iden=False,
                 local_delta_se3=False):
        super(PosVelTransitionModel, self).__init__()
        self.se3_dim = ctrlnets.get_se3_dimension(se3_type=se3_type, use_pivot=(delta_pivot == 'pred')) # Only if we are predicting directly
        self.num_se3 = num_se3

        # Pose encoder
        pdim = [256, 256, 512]
        self.poseencoder = nn.Sequential(
                                nn.Linear(2 * self.num_se3 * 12, pdim[0]), # Takes pos/vel as input
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(pdim[0], pdim[1]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(pdim[1], pdim[2]),
                                ctrlnets.get_nonlinearity(nonlinearity),
        )

        # Control encoder
        cdim = [128, 256] #[64, 128]
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, cdim[0]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                                nn.Linear(cdim[0], cdim[1]),
                                ctrlnets.get_nonlinearity(nonlinearity),
                            )

        # # Jt angle encoder
        # jdim = [0, 0]
        # self.use_jt_angles = use_jt_angles
        # if use_jt_angles:
        #     jdim = [128, 256]  # [64, 128]
        #     self.jtangleencoder = nn.Sequential(
        #         nn.Linear(num_state, jdim[0]),
        #         get_nonlinearity(nonlinearity),
        #         nn.Linear(jdim[0], jdim[1]),
        #         get_nonlinearity(nonlinearity)
        #     )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(pdim[2]+cdim[1], 512),
            ctrlnets.get_nonlinearity(nonlinearity),
            nn.Linear(512, 256),
            ctrlnets.get_nonlinearity(nonlinearity),
            nn.Linear(256, 2 * self.num_se3 * self.se3_dim)
        )

        # Initialize the SE3 decoder to predict identity SE3
        if init_se3_iden:
            print("Initializing SE3 prediction layer of the transition model to predict identity transform")
            layer = self.deltase3decoder[4]  # Get final SE3 prediction module
            ctrlnets.init_se3layer_identity(layer, num_se3, se3_type) # Init to identity

        # Create pose decoder (convert to r/t)
        self.se3_type    = se3_type
        self.delta_pivot = delta_pivot
        self.inp_pivot   = (self.delta_pivot != '') and (self.delta_pivot != 'pred') # Only for these 2 cases, no pivot is passed in as input
        self.deltaposedecoder = nn.Sequential()
        #if se3_type != 'se3aar':
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
        #if self.se3_type == 'se3aar':
        #    x = se3ToRt(x) # Use new se3ToRt layer!
        x = self.deltaposedecoder(x)  # Convert delta-SE3 to delta-Pose (can be in local or global frame of reference)
        if self.local_delta_se3:
            # Predicted delta is in the local frame of reference, can't use it directly
            z = self.posedecoder(p, x) # SE3_2 = SE3_1 * D_local (takes a point in local frame to global frame)
            y = x #TODO: This is not correct, fix it! self.globaldeltadecoder(z, self.rtinv(p)) # D_global = SE3_2 * SE3_1^-1 = SE3_1 * D_local * SE3_1^-1 (from global to global)
        else:
            # Predicted delta is already in the global frame of reference, use it directly (from global to global)
            z = self.posedecoder(x, p) # Compose predicted delta & input pose to get next pose (SE3_2 = SE3_2 * SE3_1^-1 * SE3_1)
            y = x # D = SE3_2 * SE3_1^-1 (global to global)

        # Return
        return [y, z] # Return both the deltas (in global frame) and the composed next pose
