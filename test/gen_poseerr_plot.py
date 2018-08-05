# Torch imports
import torch
import torch.nn as nn

# Global imports
import os
import sys
import shutil
import time
import numpy as np

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

######################
#### Setup options
# Parse arguments
import argparse
import configargparse
parser = configargparse.ArgumentParser(description='Train transition models on GT pose data (Baxter)')

# Dataset options
parser.add_argument('-d', '--data', required=True,
                    help='Path to tar file with pose data')


######################
def main():
    ## Parse args
    global args, num_train_iter
    args = parser.parse_args()

    ########################
    ## Load data
    if args.data.find('~') != -1:
        args.data = os.path.expanduser(args.data)
    assert (os.path.exists(args.data))

    # Get the dataset
    loaddata = torch.load(args.data)
    if 'pargs' in loaddata.keys():
        dataset = loaddata['posedata']
    else:
        dataset = loaddata

    # Get train/val/test data
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']

    # Figure out which examples belong to which joint's motions (just see which jts move)
    poses, jtangles = test_dataset['predposes'], test_dataset['jtangles']
    stids = [2]
    for j in range(2, len(jtangles)):
        _, jtid = (jtangles[j][0] - jtangles[j][-1]).abs().max(dim=0)
        if len(stids) < (jtid.item() + 1):
            stids.append(j)
    stids.append(len(jtangles))

    # Plot results
    import matplotlib.pyplot as plt
    import numpy as np
    plt.ion()

    # Compute errors in poses for errors in jt angles through the sequence
    poseerrjt, jterrjt = [], []
    poseerrnjt, jterrnjt = [], []
    for k in range(jtangles[0].size(1)):
        st, ed = stids[k], stids[k+1]
        posejt, jtanglejt = torch.stack(poses[st:ed]), torch.stack(jtangles[st:ed])
        nex, nseq = posejt.size(0), posejt.size(1)
        perr, jerr = (posejt - posejt.narrow(1,nseq-1,1).expand_as(posejt)).pow(2).view(ed-st,nseq,-1).sum(2), \
                     (jtanglejt - jtanglejt.narrow(1,nseq-1,1).expand_as(jtanglejt)).pow(2).view(ed-st,nseq,-1).sum(2)
        perrn, jerrn = perr / perr[:,0:1], jerr / jerr[:,0:1]
        poseerrjt.append(perr); poseerrnjt.append(perrn)
        jterrjt.append(jerr); jterrnjt.append(jerrn)

        # Compute mean/std statistics across the examples
        jerr_m, jerr_std = jerr.mean(dim=0), jerr.std(dim=0)
        jerrn_m, jerrn_std = jerrn.mean(dim=0), jerrn.std(dim=0)
        perr_m, perr_std   = perr.mean(dim=0), perr.std(dim=0)
        perrn_m, perrn_std = perrn.mean(dim=0), perrn.std(dim=0)

        # Subplots for err vs errn
        plt.figure(101)
        plt.subplot(121)
        plt.errorbar(np.arange(0,nseq), perr_m.numpy(), yerr=1.96*perr_std.numpy()/(nex**0.5), label="Joint-{}".format(k))
        #plt.plot(perr_m.numpy(), label="Joint-{}".format(k))
        plt.subplot(122)
        plt.errorbar(np.arange(0,nseq), perrn_m.numpy(), yerr=2*perrn_std.numpy()/(nex**0.5), label="Joint-{}".format(k))
        #plt.plot(perrn_m.numpy(), label="Joint-{}".format(k))

        # Subplots for pose vs jt angle error
        plt.figure(102)
        plt.subplot(121)
        plt.plot(jerr_m.numpy(), perr_m.numpy(), label="Joint-{}".format(k))
        plt.subplot(122)
        plt.plot(jerrn_m.numpy(), perrn_m.numpy(), label="Joint-{}".format(k))

    # Title and legend
    nticks = 6
    plt.figure(101)
    plt.subplot(121)
    plt.title("Pose error (vs) Distance to target")
    plt.xticks(np.linspace(0,nseq,nticks), [1.0,0.8,0.6,0.4,0.2,0])
    plt.legend()
    plt.subplot(122)
    plt.title("Normalized pose error (vs) Distance to target")
    plt.xticks(np.linspace(0,nseq,nticks), [1.0,0.8,0.6,0.4,0.2,0])
    plt.legend()

    # Title and legend
    plt.figure(102)
    plt.subplot(121)
    plt.title("Pose error (vs) Jt angle error")
    plt.legend()
    plt.subplot(122)
    plt.title("Normalized pose error (vs) Normalized jt angle error")
    plt.legend()

    # Show plot
    plt.show()

    ##########
    # Multiple joints
    def non_increasing(L):
        return all(x >= y for x, y in zip(L, L[1:]))

    nmax, nseq = 1000, jtangles[2].size(0)
    stt, edd = [], []
    poseerrall, jterrall = [], []
    poseerrnall, jterrnall = [], []
    while len(stt) < nmax:
        st = np.random.randint(0, len(jtangles[0])-nseq-1)
        ed = st + nseq
        posejt, jtanglejt = poses[0][st:ed], jtangles[0][st:ed]
        perr, jerr = (posejt - posejt.narrow(0, nseq-1, 1).expand_as(posejt)).pow(2).view(nseq, -1).sum(1), \
                     (jtanglejt - jtanglejt.narrow(0, nseq-1, 1).expand_as(jtanglejt)).pow(2).view(nseq, -1).sum(1)
        perrn, jerrn = perr / perr[0:1], jerr / jerr[0:1]

        # Check if the max error is > threshold and error is decreasing monotonically
        if (jerr[0].pow(0.5) < 0.5) and (jerr[0].pow(2) > 0.75):
            continue
        if not non_increasing(list(jerr.numpy())):
            continue
        stt.append(st); edd.append(ed)
        if len(stt) % 25 == 0:
            print("{}/{}".format(len(stt), nmax))

        poseerrall.append(perr)
        poseerrnall.append(perrn)
        jterrall.append(jerr)
        jterrnall.append(jerrn)

    # Compute mean/std statistics across the examples
    jerr_m, jerr_std    = torch.stack(jterrall).mean(dim=0), torch.stack(jterrall).std(dim=0)
    jerrn_m, jerrn_std  = torch.stack(jterrnall).mean(dim=0), torch.stack(jterrnall).std(dim=0)
    perr_m, perr_std    = torch.stack(poseerrall).mean(dim=0), torch.stack(poseerrall).std(dim=0)
    perrn_m, perrn_std  = torch.stack(poseerrnall).mean(dim=0), torch.stack(poseerrnall).std(dim=0)

    # Subplots for err vs errn
    plt.figure(103)
    plt.subplot(121)
    plt.errorbar(np.arange(0, nseq), perr_m.numpy(), yerr=1.96 * perr_std.numpy() / (nmax ** 0.5))
    # plt.plot(perr_m.numpy(), label="Joint-{}".format(k))
    plt.subplot(122)
    plt.errorbar(np.arange(0, nseq), perrn_m.numpy(), yerr=2 * perrn_std.numpy() / (nmax ** 0.5))
    # plt.plot(perrn_m.numpy(), label="Joint-{}".format(k))

    # Subplots for pose vs jt angle error
    plt.figure(104)
    plt.subplot(121)
    plt.plot(jerr_m.numpy(), perr_m.numpy())
    plt.subplot(122)
    plt.plot(jerrn_m.numpy(), perrn_m.numpy())

    # Title and legend
    nticks = 6
    plt.figure(103)
    plt.subplot(121)
    plt.title("Pose error (vs) Distance to target")
    plt.xticks(np.linspace(0, nseq, nticks), [1.0, 0.8, 0.6, 0.4, 0.2, 0])
    plt.legend()
    plt.subplot(122)
    plt.title("Normalized pose error (vs) Distance to target")
    plt.xticks(np.linspace(0, nseq, nticks), [1.0, 0.8, 0.6, 0.4, 0.2, 0])
    plt.legend()

    # Title and legend
    plt.figure(104)
    plt.subplot(121)
    plt.title("Pose error (vs) Jt angle error")
    plt.legend()
    plt.subplot(122)
    plt.title("Normalized pose error (vs) Normalized jt angle error")
    plt.legend()

    # Show plot
    plt.show()