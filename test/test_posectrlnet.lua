require 'torch'
require 'se3depthpred' 	
require 'nngraph'

--------------------------
-- d1. Code to choose a non-linearity
function load_nonlinearlayer(nonlinearity, options)
	local nonlinearlayer
	if nonlinearity == 'prelu' then
		nonlinearlayer = nn.PReLU()
	elseif nonlinearity == 'relu' then
		nonlinearlayer = nn.ReLU()
	elseif nonlinearity == 'tanh' then
		nonlinearlayer = nn.Tanh()
	elseif nonlinearity == 'sigmoid' then
		nonlinearlayer = nn.Sigmoid()
	elseif nonlinearity == 'elu' then
		local alpha = ((options and options.eluAlpha) or 1.0);
		print('==> Using Alpha of '..alpha..' for ELU non-linearity');
		nonlinearlayer = nn.ELU(alpha);
	else
		assert(false and "Unknown non-linearity input. Allowed are: 'prelu', 'relu', 'tanh', 'sigmoid', 'elu'");
	end
	return nonlinearlayer
end

-- d2. Create the network
function create_net(_nCtrl, _nSE3, _se3Type, _se3Dim, _slimmodel, _kinchain, _nonlinearity)	

	-- Slim model
	local slimmodel = _slimmodel or false
	if slimmodel then print("==> Using the slim version of the recurrent network"); end

	-- Kinematic chain
	local kinchain   = _kinchain or false
	if kinchain then print("==> Using the kinematic chain as part of the FK-Pose network"); end

	-- Non-linearity to use
	local nonlinearity   = _nonlinearity or 'prelu'
	local nonlinearlayer = load_nonlinearlayer(nonlinearity, options);

	------
	-- Pose prediction options
	local stateDim = _nSE3 * 12 -- R|t per SE3

	------------------------------------------------------------
	-- @@@@@@@@@@
	------------------------------
	-- Setup transition model
	-- Takes in {pose_t, ctrl_t} to predict pose_(t+1) [OR] delta_(t+1)

	---------------
	-- Setup inputs
	local idense3 = nn.Identity()();
	local pose    = nn.SelectTable(1)(idense3);
	local ctrl 	  = nn.SelectTable(2)(idense3);
	
	---------------
	-- Encode state
	local SDIMS = (slimmodel and {64, 128}) or ({128, 256});
	local R1  = nn.Reshape(-1)(pose);
	local S1  = nn.Linear(stateDim, SDIMS[1])(R1);
	local NS1 = nonlinearlayer:clone()(S1);
	local S2  = nn.Linear(SDIMS[1], SDIMS[2])(NS1);
	local NS2 = nonlinearlayer:clone()(S2);
	 	
	---------------
	-- Encode ctrl
	local CDIMS = (slimmodel and {64, 128}) or ({128, 256});
	local C1  = nn.Linear(_nCtrl, CDIMS[1])(ctrl);
	local NC1 = nonlinearlayer:clone()(C1);
	local C2  = nn.Linear(CDIMS[1], CDIMS[2])(NC1);
	local NC2 = nonlinearlayer:clone()(C2);

	---------------
	-- Concat the encoded vectors
	local CAT = nn.JoinTable(2)({NS2, NC2});

	---------------
	-- Use a decoder to predict the final SE3 output (always in the low-dim SE3 space)
	local EDIM = (slimmodel and {128, 64}) or {256, 128};
	local E1  = nn.Linear(SDIMS[2]+CDIMS[2], EDIM[1])(CAT);
	local NE1 = nonlinearlayer:clone()(E1);
	local E2  = nn.Linear(EDIM[1], EDIM[2])(NE1);
	local NE2 = nonlinearlayer:clone()(E2); -- Non-linearity 1 
	local E3  = nn.Linear(EDIM[2], _nSE3 * _se3Dim)(NE2);
	local VV  = nn.View(-1, _nSE3, _se3Dim)(E3); -- View as (B x nSE3 x se3Dim)

	---------------
	-- Use 3D positions from the input poses as the pivot point for the rotation
	local predRt 		   = nn.SE3ToRt(_se3Type, false)(VV); -- Convert the delta to Rt
	local prednextpose_1 = (_kinchain and nn.ComposeRt(false)(predRt)) or predRt; 
	local finalnextpose  = nn.ComposeRtPair()({prednextpose_1, pose}); -- SE3_2 = SE3_2 * SE3_1^-1 * SE3_1

	---------------
	-- Setup transition network
	local transitionmodel = nn.gModule({idense3}, {finalnextpose}); 
	return transitionmodel
end

-- Setup transition model
torch.manualSeed(100)
batchsize, nCtrl, nSE3, se3Type, se3Dim = 2, 10, 2, 'se3quat', 7
transmodel = create_net(nCtrl, nSE3, se3Type, se3Dim, false, false, 'prelu')

-- Create simple data and do fwd prop
pose = torch.rand(batchsize,nSE3,3,4)
ctrl = torch.rand(batchsize,nCtrl)
nextpose = transmodel:forward({pose,ctrl})
print('==> FWD pass output ==>')
print(nextpose)

-- Compute loss and do backprop
target = torch.rand(batchsize,nSE3,3,4)
losslayer = nn.MSECriterion()
loss = losslayer:forward(nextpose, target)
lossgrad = losslayer:backward(nextpose, target)
print('==> Loss: '..loss..' ==>')
print('==> Loss gradients ==>')
print(lossgrad)

-- Backprop across model
posectrlgrad = transmodel:backward({pose,ctrl}, lossgrad)
posegrad, ctrlgrad = unpack(posectrlgrad)
print('==> BWD pass gradients ==>')
print(posegrad)
print(ctrlgrad)
