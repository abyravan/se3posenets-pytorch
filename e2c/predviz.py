# Global imports
import os
import sys
import shutil
import time
import numpy as np
import random
import argparse, configargparse
import matplotlib.pyplot as plt
import json

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.distributions
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import _init_paths
import util
import e2c.model as e2cmodel
import e2c.helpers as e2chelpers

########################
############# Setup options for visualization
parser = configargparse.ArgumentParser(description='E2C/VAE prediction/visualization on block data')

# Dataset options
parser.add_argument('-c', '--config', required=True, is_config_file=True,
                    help='Path to config file for parameters')
parser.add_argument('--model-dict', default='{}', type=json.loads,
                    help='Dictionary of model names: model paths (default: {})')
parser.add_argument('--num-examples', default=100, type=int,
                    metavar='N', help='Number of examples to run tests on (default: 100)')
parser.add_argument('--seq-len', default=25, type=int,
                    metavar='S', help='Sequence length for each example (default: 25)')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='B', help='Batch size (default: 16)')
parser.add_argument('--data-key', default='val', type=str,
                        help='Run tests on this dataset: train | [val] | test')
parser.add_argument('--jt-mean-thresh', default=1e-3, type=float,
                    help='Threshold for avg. jt angle differences for an example (default: 1e-3)')
parser.add_argument('-s', '--save-dir', type=str, required=True,
                    metavar='PATH', help='Directory to save results in. If it doesnt exist, will be created. (default: )')

################ MAIN
#@profile
def main():
    # Parse args
    global vargs, num_train_iter
    vargs = parser.parse_args()

    # Create save directory and start tensorboard logger
    util.create_dir(vargs.save_dir)  # Create directory

    # Create logfile to save prints
    logfile = open(vargs.save_dir + '/logfile.txt', 'w')
    backup = sys.stdout
    sys.stdout = util.Tee(sys.stdout, logfile)

    ########################
    ############ Load models and get default options
    models, args = {}, None
    for key, val in vargs.model_dict.items():
        if os.path.isfile(val):
            ### Load checkpoint
            print("=> [{}] Loading model from checkpoint '{}'".format(key, val))
            checkpoint = torch.load(val)
            pargs = checkpoint['args']
            if args is None:
                args = checkpoint['args']

            ### Display stuff
            best_loss  = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
            best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
            print("=> [{}] Epoch {}, Train iter {}, Best validation loss: {} was from epoch: {}"
                  .format(key, checkpoint['epoch'], checkpoint['train_iter'],
                          best_loss, best_epoch))

            ### Load the model
            if pargs.deterministic:
                print('Using deterministic model')
                assert (pargs.varkl_wt == 0), "Deterministic model cannot have varkl-wt > 0"
                modelfn = e2cmodel.DeterministicModel
            else:
                print('Using probabilistic model')
                modelfn = e2cmodel.E2CModel
            model = modelfn(
                enc_img_type=pargs.enc_img_type, dec_img_type=pargs.dec_img_type,
                enc_inp_state=pargs.enc_inp_state, dec_pred_state=pargs.dec_pred_state,
                conv_enc_dec=pargs.conv_enc_dec, dec_pred_norm_rgb=pargs.dec_pred_norm_rgb,
                trans_setting=pargs.trans_setting, trans_pred_deltas=pargs.trans_pred_deltas,
                trans_model_type=pargs.trans_model_type, state_dim=pargs.num_ctrl,
                ctrl_dim=pargs.num_ctrl, wide_model=pargs.wide_model, nonlin_type=pargs.nonlin_type,
                norm_type=pargs.norm_type, coord_conv=pargs.coord_conv, img_size=(pargs.img_ht, pargs.img_wd))
            if pargs.cuda:
                model.cuda()  # Convert to CUDA if enabled

            # Print number of trainable parameters in model
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('[{}] Number of trainable parameters in model: {}'.format(key, num_params))

            # Update model params with trained checkpoint
            model.load_state_dict(checkpoint['model_state_dict']) # Load network
            models[key] = model
        else:
            print("=> [{}] No checkpoint found at '{}'".format(key, val))

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Update the sequence length in the dataset
    args.seq_len = vargs.seq_len

    # Setup datasets
    train_dataset, val_dataset, test_dataset = e2chelpers.parse_options_and_setup_block_dataset_loader(args)

    # Use dataset based on key
    if vargs.data_key == 'train':
        dataset = train_dataset
    elif vargs.data_key == 'val':
        dataset = val_dataset
    elif vargs.data_key == 'test':
        dataset = test_dataset
    else:
        assert False, "Unknown data key input: {}".format(vargs.data_key)

    # Run without gradients
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        # Load examples
        exampleids = []
        pts_f, rgbs_f, states_f, ctrls_f = [], [], [], []
        while len(exampleids) < vargs.num_examples:
            # Load sample from disk
            id = np.random.randint(len(dataset))
            sample = dataset[id]

            # Check changes to state
            statediff   = sample['states'][1:] - sample['states'][:-1]
            meanjtdiff  = statediff[:,:-1].abs().mean(1).gt(vargs.jt_mean_thresh)
            meanjtper   = meanjtdiff.float().sum().item() / statediff.size(0) # % of sequence with value > 1e-3
            gripdiff    = statediff[:,-1].abs().gt(5e-4)
            gripnum     = gripdiff.float().sum().item()

            # Threshold to get good examples
            if (meanjtper > 0.85) or (gripnum > 2):
                exampleids.append(id)
                if (len(exampleids) % 10) == 0:
                    print('Loaded example {}/{}'.format(len(exampleids), vargs.num_examples))

                # Get inputs and targets (B x S x C x H x W), (B x S x NDIM)
                pts_f.append(sample['points'])
                rgbs_f.append(sample['rgbs'].type_as(sample['points']) / 255.0)  # Normalize RGB to 0-1
                states_f.append(sample['states'])  # Joint angles, gripper states
                ctrls_f.append(sample['controls'])  # Controls
            else:
                print('Discarding. % mean jt angle diff > {} is {}/0.85. Gripper num: {}/2'.format(
                    vargs.jt_mean_thresh, meanjtper, gripnum))

        # Stack data
        pts_f, rgbs_f, states_f, ctrls_f = torch.stack(pts_f, 0), torch.stack(rgbs_f, 0), \
                                           torch.stack(states_f, 0), torch.stack(ctrls_f, 0)
        print(pts_f.size())
        
        # Setup image inputs/outputs based on provided type (B x S x C x H x W)
        inputimgs_f  = e2chelpers.concat_image_data(pts_f, rgbs_f, args.enc_img_type)
        outputimgs_f = e2chelpers.concat_image_data(pts_f, rgbs_f, args.dec_img_type)

        # ============ FWD pass + Compute loss ============#
        # Get the ids for running forward pass
        idseq = list(np.arange(0, vargs.num_examples, vargs.batch_size))
        if idseq[-1] != vargs.num_examples:
            idseq.append(vargs.num_examples)
        bsz, nex, seq = vargs.batch_size, vargs.num_examples, pts_f.size(1)

        ####### Run a forward pass through the networks for predictions
        stats = {}
        for key, model in models.items():
            # Create things to save stuff
            print('Running forward pass through the model: {}'.format(key))
            MST = {'encimgerrs'    : torch.zeros(nex, seq),
                   'transimgerrs'  : torch.zeros(nex, seq-1),
                   'encimgerrs_t'  : torch.zeros(nex, seq),
                   'transimgerrs_t': torch.zeros(nex, seq-1),
                   'transsterrs'   : torch.zeros(nex, seq-1),
                   'transsterrs_t' : torch.zeros(nex, seq-1),
                   'encsterrs_t'   : torch.zeros(nex, seq),
                   'transdterrs'   : torch.zeros(nex, seq-1),
                   'transdterrs_t' : torch.zeros(nex, seq-1),
                   'encdterrs_t'   : torch.zeros(nex, seq)}

            # Iterate over the examples
            for j in range(len(idseq)-1):
                # Choose the examples
                st, ed = idseq[j], idseq[j+1]

                # Push to CUDA
                states, ctrls         = states_f[st:ed].to(device), ctrls_f[st:ed].to(device)
                inputimgs, outputimgs = inputimgs_f[st:ed].to(device), outputimgs_f[st:ed].to(device)

                ### Run through the model
                if args.deterministic:
                    encstates, transstates, decimgs = \
                        model.forward(inputimgs, states, ctrls)
                    encsamples, transsamples = torch.cat(encstates, 1), torch.cat(transstates, 1) # B x S x H
                else:
                    encdists, encsamples, transdists, transsamples, decimgs = \
                        model.forward(inputimgs, states, ctrls)
                    encsamples, transsamples = torch.cat(encsamples, 1), torch.cat(transsamples, 1)
                print(encsamples.size(), transsamples.size())

                ### Get the images from the encoder states & transition model states
                encdecimgs, transdecimgs = [decimgs[0]], decimgs[1:]
                for k in range(seq-1):
                    decimg = model.decoder.forward(encsamples[:,k+1])
                    encdecimgs.append(decimg)
                encdecimgs, transdecimgs = torch.stack(encdecimgs, 1), torch.stack(transdecimgs, 1) # B x S x C x H x W

                ### Measure image errors between encdecimgs/outputimgs, transdecimgs/outputimgs
                MST['encimgerrs'][st:ed]   = (encdecimgs - outputimgs).pow(2).view(bsz, seq, -1).mean(2).cpu()
                MST['transimgerrs'][st:ed] = (transdecimgs - outputimgs[:,1:]).pow(2).view(bsz, seq-1, -1).mean(2).cpu()

                ### Measure image errors to target image
                tarimg = outputimgs[:,-1:]
                MST['encimgerrs_t'][st:ed] = (encdecimgs - tarimg.expand_as(encdecimgs)).pow(2).view(bsz, seq, -1).mean(2).cpu()
                MST['transimgerrs'][st:ed] = (transdecimgs - tarimg.expand_as(transdecimgs)).pow(2).view(bsz, seq-1, -1).mean(2).cpu()

                ### Measure errors between encstates/transstates
                tarsample           = encsamples[:,-1:]
                MST['transsterrs'][st:ed]   = (encsamples[:,1:] - transsamples).pow(2).view(bsz, seq-1, -1).mean(2).cpu()
                MST['transsterrs_t'][st:ed] = (tarsample.expand_as(transsamples) - transsamples).pow(2).view(bsz, seq-1, -1).mean(2).cpu()
                MST['encsterrs_t'][st:ed]   = (tarsample.expand_as(encsamples) - encsamples).pow(2).view(bsz, seq, -1).mean(2).cpu()

                ### Distributional errors (if present)
                if not args.deterministic:
                    transdisterrs, transdisterrs_tar, encdisterrs_tar = [], [], []
                    for k in range(seq):
                        # Transition vs encoder errors
                        if (k < seq-1):
                            # Trans errs
                            if transdists[k] is None:
                                transdisterr     = (transsamples[:,k] - encdists[k+1].mean).pow(2).view(bsz,-1).mean(1)
                                transdisterr_tar = (transsamples[:,k] - encdists[-1].mean).pow(2).view(bsz,-1).mean(1)
                            else:
                                transdisterr     = torch.distributions.kl.kl_divergence(transdists[k], encdists[k+1]).view(bsz)
                                transdisterr_tar = torch.distributions.kl.kl_divergence(transdists[k], encdists[-1]).view(bsz)
                            transdisterrs.append(transdisterr) # B x 1
                            transdisterrs_tar.append(transdisterr_tar)  # B x 1

                        # Encoder vs target enc dist errs
                        encdisterr_tar = torch.distributions.kl.kl_divergence(encdists[k], encdists[-1]).view(bsz)
                        encdisterrs_tar.append(encdisterr_tar)

                    # Stack the data
                    MST['transdterrs'][st:ed]   = torch.stack(transdisterrs, 1).cpu()     # B x S
                    MST['transdterrs_t'][st:ed] = torch.stack(transdisterrs_tar, 1).cpu() # B x S
                    MST['encdterrs_t'][st:ed]   = torch.stack(encdisterrs_tar, 1).cpu()   # B x S

                ### todo: Save some image sequences (keep those in memory?)


            ### Save stats
            stats[key] = MST

        # todo: print some summary stats for each model
        # todo: create some matplotlib plots from these stats
        plkeys = [key for key, _ in stats[next(iter(stats))].items()]
        for plkey in range(len(plkeys)):
            # Setup plot figure
            plt.figure(100+plkey)
            plt.hold(True)
            for key, MST in stats.items():
                mean, std = MST[plkey].mean(0).numpy(), MST[plkey].std(0).numpy()
                plt.plot(mean, label=key)
            plt.legend()
            plt.title(plkey)

        ### Save stuff to disk now
        torch.save({'stats': stats,
                    'vargs': vargs,
                    'exids': exampleids}, vargs.save_dir + "/predvizstats.pth.tar")

################ RUN MAIN
if __name__ == '__main__':
    main()