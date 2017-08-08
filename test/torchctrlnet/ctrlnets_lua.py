import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
import se3layers as se3nn
from torch.autograd import Variable

def copy_conv_params(layer, params):
    layer.weight.data.copy_(params.weight)
    layer.bias.data.copy_(params.bias)

def copy_bn_params(layer, params):
    layer.running_mean.copy_(params.running_mean)
    layer.running_var.copy_(params.running_var)
    layer.weight.data.copy_(params.weight)
    layer.bias.data.copy_(params.bias)

def copy_fc_params(layer, params):
    layer.weight.data.copy_(params.weight)
    layer.bias.data.copy_(params.bias)

def copy_prelu_params(layer, params):
    layer.weight.data.copy_(params.weight)

### Pose-Mask Encoder
# Model that takes in "depth/point cloud" to generate "k"-channel masks and "k" poses represented as [R|t]
class PoseMaskEncoder(nn.Module):
    def __init__(self, params=None):
        super(PoseMaskEncoder, self).__init__()

        self.num_se3   = 8
        se3_type  = 'se3aa'
        self.se3_dim   = 6
        use_pivot = False

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        self.c1 = nn.Conv2d(3, 8, kernel_size=9, stride=1, padding=4)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1 = nn.BatchNorm2d(8)
        self.n1 = nn.PReLU()

        self.c2 = nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=3)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b2 = nn.BatchNorm2d(16)
        self.n2 = nn.PReLU()

        self.c3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b3 = nn.BatchNorm2d(32)
        self.n3 = nn.PReLU()

        self.c4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b4 = nn.BatchNorm2d(64)
        self.n4 = nn.PReLU()

        self.c5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.p5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b5 = nn.BatchNorm2d(128)
        self.n5 = nn.PReLU()

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.c6 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0) # 1x1, 7x10 -> 7x10
        self.b6 = nn.BatchNorm2d(128)
        self.n6 = nn.PReLU()

        self.d1 = nn.ConvTranspose2d(128, 64, kernel_size=(3,4), stride=2, padding=(0,1))
        self.bd1 = nn.BatchNorm2d(64)
        self.nd1 = nn.PReLU()

        self.d2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bd2 = nn.BatchNorm2d(32)
        self.nd2 = nn.PReLU()

        self.d3 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=2)
        self.bd3 = nn.BatchNorm2d(16)
        self.nd3 = nn.PReLU()

        self.d4 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2, padding=2)
        self.bd4 = nn.BatchNorm2d(8)
        self.nd4 = nn.PReLU()

        self.d5 = nn.ConvTranspose2d(8, self.num_se3, kernel_size=8, stride=2, padding=3)  # 8x8, 120x160 -> 240x320

        # Normalize to generate mask (wt-sharpening vs soft-mask model)
        self.maskdecoder = nn.Softmax2d() # SoftMax normalization

        ###### Pose Decoder
        # Create SE3 decoder (take conv output, reshape, run FC layers to generate "num_se3" poses)
        self.se3decoder  = nn.Sequential(
                                nn.Linear(128*7*10, 64),
                                nn.PReLU(),
                                nn.Linear(64, self.num_se3 * self.se3_dim), # Predict the SE3s from the conv-output
                           )

        self.posedecoder = se3nn.SE3ToRt(se3_type, use_pivot)

        ## Copy weights from the parameters (lua)
        if params is not None:
            # Encoder params
            copy_conv_params(self.c1, params[1])
            copy_bn_params(self.b1, params[3])
            copy_prelu_params(self.n1, params[4])

            copy_conv_params(self.c2, params[5])
            copy_bn_params(self.b2, params[7])
            copy_prelu_params(self.n2, params[8])

            copy_conv_params(self.c3, params[9])
            copy_bn_params(self.b3, params[11])
            copy_prelu_params(self.n3, params[12])

            copy_conv_params(self.c4, params[13])
            copy_bn_params(self.b4, params[15])
            copy_prelu_params(self.n4, params[16])

            copy_conv_params(self.c5, params[17])
            copy_bn_params(self.b5, params[19])
            copy_prelu_params(self.n5, params[20])

            # Pose decoder params
            copy_fc_params(self.se3decoder[0], params[22])
            copy_prelu_params(self.se3decoder[1], params[23])
            copy_fc_params(self.se3decoder[2], params[24])

            # Deconv decoder params
            copy_conv_params(self.c6, params[27])
            copy_bn_params(self.b6, params[28])
            copy_prelu_params(self.n6, params[29])

            copy_conv_params(self.d1, params[30])
            copy_bn_params(self.bd1, params[32])
            copy_prelu_params(self.nd1, params[33])

            copy_conv_params(self.d2, params[34])
            copy_bn_params(self.bd2, params[36])
            copy_prelu_params(self.nd2, params[37])

            copy_conv_params(self.d3, params[38])
            copy_bn_params(self.bd3, params[40])
            copy_prelu_params(self.nd3, params[41])

            copy_conv_params(self.d4, params[42])
            copy_bn_params(self.bd4, params[44])
            copy_prelu_params(self.nd4, params[45])

            copy_conv_params(self.d5, params[46])

    # Run forward pass
    def forward(self, x, train_iter=0):
        # Run conv-encoder to generate embedding
        c1 = self.n1(self.b1(self.p1(self.c1(x))))
        c2 = self.n2(self.b2(self.p2(self.c2(c1))))
        c3 = self.n3(self.b3(self.p3(self.c3(c2))))
        c4 = self.n4(self.b4(self.p4(self.c4(c3))))
        c5 = self.n5(self.b5(self.p5(self.c5(c4))))

        # Run mask-decoder to predict a smooth mask
        m = self.n6(self.b6(self.c6(c5)))
        m = self.nd1(self.bd1(self.d1(m) + c4))
        m = self.nd2(self.bd2(self.d2(m) + c3))
        m = self.nd3(self.bd3(self.d3(m) + c2))
        m = self.nd4(self.bd4(self.d4(m) + c1))
        m = self.d5(m)

        m = self.maskdecoder(m)

        # Run pose-decoder to predict poses
        p = c5.view(-1, 128*7*10)
        p = self.se3decoder(p)
        p = p.view(-1, self.num_se3, self.se3_dim)
        p = self.posedecoder(p)

        # Return poses and masks
        return [p, m]

### Transition model
# Takes in [pose_t, ctrl_t] and generates delta pose between t & t+1
class TransitionModel(nn.Module):
    def __init__(self, params):
        super(TransitionModel, self).__init__()
        self.num_se3 = 8
        se3_type = 'se3aa'
        self.se3_dim = 6
        use_pivot = False
        num_ctrl = 7

        # Pose encoder
        self.poseencoder = nn.Sequential(
                                nn.Linear(self.num_se3 * 12, 128),
                                nn.PReLU(),
                                nn.Linear(128, 256),
                                nn.PReLU()
                            )

        # Control encoder
        self.ctrlencoder = nn.Sequential(
                                nn.Linear(num_ctrl, 128),
                                nn.PReLU(),
                                nn.Linear(128, 256),
                                nn.PReLU()
                            )

        # SE3 decoder
        self.deltase3decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, self.num_se3 * self.se3_dim)
        )

        # Create pose decoder (convert to r/t)
        self.deltaposedecoder = se3nn.SE3ToRt(se3_type, use_pivot)  # Convert to Rt

        # Compose deltas with prev pose to get next pose
        # It predicts the delta transform of the object between time "t1" and "t2": p_t2_to_t1: takes a point in t2 and converts it to t1's frame of reference
	    # So, the full transform is: p_t2 = p_t1 * p_t2_to_t1 (or: p_t2 (t2 to cam) = p_t1 (t1 to cam) * p_t2_to_t1 (t2 to t1))
        self.posedecoder = se3nn.ComposeRtPair()

        ## Copy weights from the parameters (lua)
        if params is not None:
            # Pose encoder
            copy_fc_params(self.poseencoder[0], params[3])
            copy_prelu_params(self.poseencoder[1], params[4])
            copy_fc_params(self.poseencoder[2], params[5])
            copy_prelu_params(self.poseencoder[3], params[6])

            # Ctrl encoder
            copy_fc_params(self.ctrlencoder[0], params[8])
            copy_prelu_params(self.ctrlencoder[1], params[9])
            copy_fc_params(self.ctrlencoder[2], params[10])
            copy_prelu_params(self.ctrlencoder[3], params[11])

            # SE3 decoder
            copy_fc_params(self.deltase3decoder[0], params[13])
            copy_prelu_params(self.deltase3decoder[1], params[14])
            copy_fc_params(self.deltase3decoder[2], params[15])
            copy_prelu_params(self.deltase3decoder[3], params[16])
            copy_fc_params(self.deltase3decoder[4], params[17])

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

class SE3PoseModel(nn.Module):
    def __init__(self):
        super(SE3PoseModel, self).__init__()
        # Models
        posemaskdata = load_lua('/home/barun/Projects/se3nets-pytorch/test/torchctrlnet/posemaskmodules.t7')
        self.posemaskmodel = PoseMaskEncoder(params=posemaskdata)

        transitiondata = load_lua('/home/barun/Projects/se3nets-pytorch/test/torchctrlnet/transitionmodules.t7')
        self.transitionmodel = TransitionModel(params=transitiondata)

    def forward(self, x):
        print 'Not implemented'
        return NotImplementedError

def main():
    # Models
    posemaskdata = load_lua('/home/barun/Projects/se3nets-pytorch/test/torchctrlnet/posemaskmodules.t7')
    posemasknet = PoseMaskEncoder(params = posemaskdata)
    posemasknet.cuda(); posemasknet.eval()
    transitiondata = load_lua('/home/barun/Projects/se3nets-pytorch/test/torchctrlnet/transitionmodules.t7')
    transitionnet = TransitionModel(params = transitiondata)
    transitionnet.cuda(); posemasknet.eval()

    # Test once
    torch.manual_seed(100)  # seed
    ptclouds = Variable(torch.rand(2,3,240,320).cuda(), requires_grad=True)
    ctrls    = Variable(torch.rand(2,7).cuda(), requires_grad=True)
    print 'Ptclouds: ', ptclouds.data.max(), ptclouds.data.min(), ptclouds.data.clone().abs().mean()
    print 'Ctrls: ', ctrls.data.max(), ctrls.data.min(), ctrls.data.clone().abs().mean()

    poses, masks = posemasknet(ptclouds)
    deltaposes, nextposes = transitionnet([poses, ctrls])
    print 'Poses: ', poses.data.max(), poses.data.min(), poses.data.clone().abs().mean()
    print 'Masks: ', masks.data.max(), masks.data.min(), masks.data.clone().abs().mean()
    print 'Next: ', nextposes.data.max(), nextposes.data.min(), nextposes.data.clone().abs().mean()
    print 'Delta: ', deltaposes.data.max(), deltaposes.data.min(), deltaposes.data.clone().abs().mean()

################ RUN MAIN
if __name__ == '__main__':
    main()

