import torch
import numpy as np
import csv
import os

# Read baxter state files
def read_baxter_state_file(filename):
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        ret['actjtpos']     = torch.Tensor(next(spamreader)[0:-1])  # Last element is a string due to the way the file is created
        ret['actjtvel']     = torch.Tensor(next(spamreader)[0:-1])
        ret['actjteff']     = torch.Tensor(next(spamreader)[0:-1])
        ret['comjtpos']     = torch.Tensor(next(spamreader)[0:-1])
        ret['comjtvel']     = torch.Tensor(next(spamreader)[0:-1])
        ret['comjtacc']     = torch.Tensor(next(spamreader)[0:-1])
        ret['tarendeffpos'] = torch.Tensor(next(spamreader)[0:-1])
        try:
            trackdata = next(spamreader)[0:-1]
            ret['trackerjtpos'] = torch.Tensor(trackdata if trackdata[-1] != '' else trackdata[:-1])
            ret['timestamp']    = next(spamreader)[0]
        except:
            ret['trackerjtpos'] = None
            ret['timestamp']    = None
    return ret


# Read baxter SE3 state file for all the joints
def read_baxter_se3state_file(filename):
    # Read all the lines in the SE3-state file
    lines = []
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile,  delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in spamreader:
            if len(row) == 0:
                continue
            if type(row[-1]) == str:  # In case we have a string at the end of the list
                row = row[0:-1]
            lines.append(torch.Tensor(row))

    # Parse the SE3-states
    ret, ctr = {}, 0
    while (ctr < len(lines)):
        id = int(lines[ctr][0])  # ID of mesh
        data = lines[ctr + 1].view(3, 4)  # Transform data
        T = torch.eye(4)
        T[0:3, 0:3] = data[0:3, 1:4]; T[0:3, 3] = data[0:3, 0]
        ret[id] = T # Add to list of transforms
        ctr += 2  # Increment counter
    return ret

# Read baxter joint labels and their corresponding mesh index value
def read_statelabels_file(filename):
    ret = {}
    with open(filename, 'rt') as csvfile:
        spamreader      = csv.reader(csvfile, delimiter=' ')
        ret['frames']   = next(spamreader)[0:-1]
        ret['meshIds']  = torch.IntTensor([int(x) for x in next(spamreader)[0:-1]])
    return ret

###########################
## Load data
path = '/raid/barun/data/baxter/multijtdata/baxter_babbling_rarm_3.5hrs_Dec14_16/postprocessmotions_f/motion0/'

## Read stats
statsfilename = os.path.join(path, 'postprocessstats.txt')
assert (os.path.exists(statsfilename))
max_flow_step = 30  # This is the maximum future step (k) for which we need flows (flow_k/)
with open(statsfilename, 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
    nexamples = int(next(reader)[0]) - max_flow_step # We only have flows for these many images!

## Read mesh ids
baxter_labels = read_statelabels_file(path + '/../statelabels.txt')
mesh_ids      = baxter_labels['meshIds']

## Iterate over examples
ndof = 7
jtangles = torch.zeros(nexamples, ndof)
poses    = torch.zeros(nexamples, len(mesh_ids), 4, 4)
for k in range(nexamples):
    # Get config
    statepath = path + '/state{}.txt'.format(k)
    state = read_baxter_state_file(statepath)
    jtangles[k] = state['actjtpos'] # state dimension

    # Load SE3 state & get all poses
    se3statepath = path + '/se3state{}.txt'.format(k)
    se3state = read_baxter_se3state_file(se3statepath)
    for j in xrange(len(mesh_ids)):
        meshid = mesh_ids[j]
        poses[k][j] = se3state[meshid] # 4 x 4 transform

## Save stuff
np.savez("jtangleposes.npz", jtangles=jtangles.numpy(), poses=poses.numpy())