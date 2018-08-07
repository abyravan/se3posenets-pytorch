# Global imports
import os
import sys
import shutil
import time
import numpy as np
import random
import argparse

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

#### Setup options
parser = e2chelpers.setup_common_options()

################ MAIN
#@profile
def main():
    # Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # Create save directory and start tensorboard logger
    util.create_dir(args.save_dir)  # Create directory
    now = time.strftime("%c")
    tblogger = util.TBLogger(args.save_dir + '/logs/' + now)  # Start tensorboard logger

    # Create logfile to save prints
    logfile = open(args.save_dir + '/logs/' + now + '/logfile.txt', 'w')
    backup = sys.stdout
    sys.stdout = util.Tee(sys.stdout, logfile)

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup datasets
    train_dataset, val_dataset, test_dataset = e2chelpers.parse_options_and_setup_block_dataset_loader(args)

    # Create a data-collater for combining the samples of the data into batches along with some post-processing
    if args.evaluate:
        # Load only test loader
        args.imgdisp_freq = 10 * args.disp_freq  # Tensorboard log frequency for the image data
        sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
        test_loader = util.DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=args.num_workers, sampler=sampler,
                                                          collate_fn=test_dataset.collate_batch))
    else:
        # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
        train_loader = util.DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers,
                                                           collate_fn=train_dataset.collate_batch))
        val_loader = util.DataEnumerator(util.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_workers,
                                                         collate_fn=val_dataset.collate_batch))

    ### Load the model
    num_train_iter = 0
    model = e2cmodel.E2CModel(
        enc_img_type=args.enc_img_type, dec_img_type=args.dec_img_type,
        enc_inp_state=args.enc_inp_state, dec_pred_state=args.dec_pred_state,
        conv_enc_dec=args.conv_enc_dec, dec_pred_norm_rgb=args.dec_pred_norm_rgb,
        trans_setting=args.trans_setting, trans_pred_deltas=args.trans_pred_deltas,
        trans_model_type=args.trans_model_type, state_dim=args.num_ctrl,
        ctrl_dim=args.num_ctrl, wide_model=args.wide_model, nonlin_type=args.nonlin_type,
        norm_type=args.norm_type, coord_conv=args.coord_conv)
    if args.cuda:
        model.cuda()  # Convert to CUDA if enabled

    # Print number of trainable parameters in model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters in model: {}'.format(num_params))

    ### Load optimizer
    optimizer = e2chelpers.load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)

    ### Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # Load checkpoint & update model/optimizer params
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            num_train_iter = checkpoint['train_iter']
            model.load_state_dict(checkpoint['model_state_dict']) # Load network
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Load optimizer
            print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
                  .format(args.resume, checkpoint['epoch'], num_train_iter))
            # Load loss
            best_loss  = checkpoint['best_loss'] if 'best_loss' in checkpoint else float("inf")
            best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
            print('==== Best validation loss: {} was from epoch: {} ===='.format(best_loss, best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_loss, best_epoch = float("inf"), 0

    ########################
    ############ Test (don't create the data loader unless needed, creates 4 extra threads)
    if args.evaluate:
        print('==== Evaluating pre-trained network on test data ===')
        test_stats = iterate(test_loader, model, tblogger, len(test_loader), mode='test')

        # Save final test error
        e2chelpers.save_checkpoint({
            'args': args,
            'test_stats': {'stats': test_stats,
                           'niters': test_loader.niters, 'nruns': test_loader.nruns,
                           'totaliters': test_loader.iteration_count(),
                           'ids': test_stats.data_ids,
                           },
        }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

        # Close log file & return
        logfile.close()
        return

    ## Create a file to log different validation errors over training epochs
    statstfile = open(args.save_dir + '/epochtrainstats.txt', 'w')
    statsvfile = open(args.save_dir + '/epochvalstats.txt', 'w')
    statstfile.write("Epoch, Loss\n")
    statsvfile.write("Epoch, Loss\n")

    ########################
    ############ Train / Validate
    args.imgdisp_freq = 5 * args.disp_freq # Tensorboard log frequency for the image data
    train_ids, val_ids = [], []
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        e2chelpers.adjust_learning_rate(optimizer, args, epoch, args.lr_decay, args.decay_epochs, args.min_lr)

        # Train for one epoch
        train_stats = iterate(train_loader, model, tblogger, args.train_ipe,
                           mode='train', optimizer=optimizer, epoch=epoch+1)
        train_ids += train_stats.data_ids

        # Evaluate on validation set
        val_stats = iterate(val_loader, model, tblogger, args.val_ipe,
                            mode='val', epoch=epoch+1)
        val_ids += val_stats.data_ids

        # Find best losses
        val_loss   = val_stats.loss.avg
        is_best    = (val_loss < best_loss)
        prev_best_loss, prev_best_epoch = best_loss, best_epoch
        s = 'SAME'
        if is_best:
            best_loss, best_epoch, s       = val_loss, epoch+1, 'IMPROVED'
        print('==== [LOSS]   Epoch: {}, Status: {}, Previous best: {:.5f}/{}. Current: {:.5f}/{} ===='.format(
                                    epoch+1, s, prev_best_loss, prev_best_epoch, best_loss, best_epoch))

        # Write losses to stats file
        statstfile.write("{}, {}\n".format(epoch+1, train_stats.loss.avg))
        statsvfile.write("{}, {}\n".format(epoch+1, val_stats.loss.avg))

        # Save checkpoint
        e2chelpers.save_checkpoint({
            'epoch': epoch+1,
            'args' : args,
            'best_loss'            : best_loss,
            'best_epoch'           : best_epoch,
            'train_stats': {'stats': train_stats,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count(),
                            'ids': train_ids,
                            },
            'val_stats'  : {'stats': val_stats,
                            'niters': val_loader.niters, 'nruns': val_loader.nruns,
                            'totaliters': val_loader.iteration_count(),
                            'ids': val_ids,
                            },
            'train_iter' : num_train_iter,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, is_best, savedir=args.save_dir, filename='checkpoint.pth.tar')
        print('\n')

    # Delete train and val data loaders
    del train_loader, val_loader

    # Load best model for testing (not latest one)
    print("=> loading best model from '{}'".format(args.save_dir + "/model_best.pth.tar"))
    checkpoint = torch.load(args.save_dir + "/model_best.pth.tar")
    num_train_iter = checkpoint['train_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded best checkpoint (epoch {}, train iter {})"
          .format(checkpoint['epoch'], num_train_iter))
    best_epoch   = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0
    print('==== Best validation loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))

    # Do final testing (if not asked to evaluate)
    # (don't create the data loader unless needed, creates 4 extra threads)
    print('==== Evaluating trained network on test data ====')
    args.imgdisp_freq = 10 * args.disp_freq # Tensorboard log frequency for the image data
    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    test_loader = util.DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_workers, sampler=sampler,
                                                      collate_fn=test_dataset.collate_batch))
    test_stats = iterate(test_loader, model, tblogger, len(test_loader),
                         mode='test', epoch=args.epochs)
    print('==== Best validation loss: {:.5f} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))

    # Save final test error
    e2chelpers.save_checkpoint({
        'args': args,
        'test_stats': {'stats': test_stats,
                       'niters': test_loader.niters, 'nruns': test_loader.nruns,
                       'totaliters': test_loader.iteration_count(),
                       'ids': test_stats.data_ids,
                       },
    }, is_best=False, savedir=args.save_dir, filename='test_stats.pth.tar')

    # Write test stats to val stats file at the end
    statsvfile.write("{}, {}\n".format(checkpoint['epoch'], test_stats.loss.avg))
    statsvfile.close(); statstfile.close()

    # Close log file
    logfile.close()

################# HELPER FUNCTIONS

### Main iterate function (train/test/val)
def iterate(data_loader, model, tblogger, num_iters,
            mode='test', optimizer=None, epoch=0):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = util.AverageMeter(), util.AverageMeter(), \
                                               util.AverageMeter(), util.AverageMeter()

    # Save all stats into a namespace
    stats = argparse.Namespace()
    stats.loss, stats.reconsloss                = util.AverageMeter(), util.AverageMeter()
    stats.varklloss, stats.transencklloss       = util.AverageMeter(), util.AverageMeter()
    stats.data_ids = []

    # Switch model modes
    train = (mode == 'train')
    if train:
        assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
        model.train()
    else:
        assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}"+mode
        model.eval()

    # Setup loss weights & functions
    recons_wt, varkl_wt = args.recons_wt * args.loss_scale, args.varkl_wt * args.loss_scale
    transenckl_wt       = args.transenckl_wt * args.loss_scale
    recons_loss_fn      = e2chelpers.get_loss_function(args.recons_loss_type)

    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(
        mode, epoch, num_iters))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(num_iters):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        j, sample = data_loader.next()
        stats.data_ids.append(sample['id'].clone())

        # Get inputs and targets (B x S x C x H x W), (B x S x NDIM)
        pts      = util.req_grad(sample['points'].to(device), train) # Need gradients
        rgbs     = util.req_grad(sample['rgbs'].type_as(pts) / 255.0, train)  # Normalize RGB to 0-1
        states   = util.req_grad(sample['states'].to(device), train) # Joint angles, gripper states
        ctrls    = util.req_grad(sample['controls'].to(device), train) # Controls

        # Setup image inputs/outputs based on provided type (B x S x C x H x W)
        inputimgs  = e2chelpers.concat_image_data(pts, rgbs, args.enc_img_type)
        outputimgs = e2chelpers.concat_image_data(pts, rgbs, args.dec_img_type).detach() # No gradients needed w.r.t these

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        ### Run a forward pass through the network for predictions
        encdists, encsamples, transdists, transsamples, decimgs = \
            model.forward(inputimgs, states, ctrls)

        ### Compute losses
        loss, reconsloss, varklloss, transencklloss = 0, torch.zeros(args.seq_len+1), \
                              torch.zeros(args.seq_len+1), torch.zeros(args.seq_len)
        for k in range(args.seq_len+1):
            # Reconstruction loss between decoded images & true images
            currreconsloss = recons_wt * recons_loss_fn(decimgs[k], outputimgs[:,k]) # Output imgs are B x (S+1)
            reconsloss[k]  = currreconsloss.item()

            # Variational KL loss between encoder distribution & standard normal distribution
            currvarklloss  = varkl_wt * e2chelpers.variational_mvnormal_kl_loss(encdists[k])
            varklloss[k]   = currvarklloss.item()

            # KL loss between encoder predictions @ t+1 & transition model predictions @ t+1
            # If transition model predicts the next sample, compute error between sample & mean of encoder distribution
            if (k < args.seq_len):
                if transdists[k] is None:
                    currtransencklloss = transenckl_wt * (transsamples[k] - encdists[k+1].mean).pow(2).mean()
                else:
                    currtransencklloss = transenckl_wt * torch.distributions.kl.kl_divergence(transdists[k],
                                                                                              encdists[k+1]).mean()
                transencklloss[k] = currtransencklloss.item()
            else:
                currtransencklloss = 0

            # Append to total loss
            loss += currreconsloss + currvarklloss + currtransencklloss

        # Update stats
        stats.reconsloss.update(reconsloss)
        stats.varklloss.update(varklloss)
        stats.transencklloss.update(transencklloss)
        stats.loss.update(loss.item())

        # Measure FWD time
        fwd_time.update(time.time() - start)

        # ============ Gradient backpass + Optimizer step ============#
        # Compute gradient and do optimizer update step (if in training mode)
        if (train):
            # Start timer
            start = time.time()

            # Backward pass & optimize
            optimizer.zero_grad()  # Zero gradients
            loss.backward()        # Compute gradients - BWD pass
            optimizer.step()       # Run update step

            # Increment number of training iterations by 1
            num_train_iter += 1

            # Measure BWD time
            bwd_time.update(time.time() - start)

        # ============ Visualization ============#
        # Make sure to not add to the computation graph (will memory leak otherwise)!
        with torch.no_grad():

            # Start timer
            start = time.time()

            # Display/Print frequency
            bsz = pts.size(0)
            if i % args.disp_freq == 0:
                ### Print statistics
                print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                            samplecurr=j+1, sampletotal=len(data_loader),
                            stats=stats, bsz=bsz)

                ### Print time taken
                print('\tTime => Data: {data.val:.3f} ({data.avg:.3f}), '
                      'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
                      'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
                      'Viz: {viz.val:.3f} ({viz.avg:.3f})'.format(
                    data=data_time, fwd=fwd_time, bwd=bwd_time, viz=viz_time))

                ### TensorBoard logging
                # (1) Log the scalar values
                iterct = data_loader.iteration_count()  # Get total number of iterations so far
                info = {
                    mode + '-loss'          : loss.item(),
                    mode + '-reconsloss'    : reconsloss.sum() / bsz,
                    mode + '-varklloss'     : varklloss.sum() / bsz,
                    mode + '-transencklloss': transencklloss.sum() / bsz,
                }
                if mode == 'train':
                    info[mode + '-lr'] = args.curr_lr  # Plot current learning rate
                for tag, value in info.items():
                    tblogger.scalar_summary(tag, value, iterct)

                # (2) Log images & print predicted SE3s
                if i % args.imgdisp_freq == 0:

                    ## Log the images (at a lower rate for now)
                    id = random.randint(0, pts.size(0)-1)

                    # Get predicted RGB, depth, pts
                    predrgbs, preddepths, predpts = [], [], []
                    for k in range(args.seq_len+1):
                        predrgb, preddepth, predpt = e2chelpers.split_image_data(decimgs[k].narrow(0,id,1),
                                                                                 args.dec_img_type, split_dim=1)
                        predrgbs.append(predrgb) # 1 x 3 x ht x wd
                        preddepths.append(preddepth)
                        predpts.append(predpt)

                    # Get GT RGB, depth, pts & concat with predicted data (if they exist)
                    imginfo = {}
                    if predrgbs[0] is not None:
                        predrgbs = torch.cat(predrgbs, 0).cpu().float() # (S+1) x 3 x ht x wd
                        gtrgbs   = rgbs[id].cpu().float()               # (S+1) x 3 x ht x wd
                        catrgbs  = torchvision.utils.make_grid(
                            torch.cat([predrgbs, gtrgbs],0).view(-1,3,args.img_ht,args.img_wd),
                            nrow=args.seq_len+1, normalize=True, range=(0.0, 1.0))
                        imginfo[mode+'-rgbs'] = util.to_np(catrgbs.unsqueeze(0))

                    if preddepths[0] is not None:
                        preddepths = torch.cat(preddepths, 0).cpu().float() # (S+1) x 1 x ht x wd
                        gtdepths   = pts[id,:,2:].cpu().float()             # (S+1) x 1 x ht x wd
                        catdepths  = torchvision.utils.make_grid(
                            torch.cat([preddepths, gtdepths],0).view(-1,1,args.img_ht,args.img_wd).expand(
                                (args.seq_len+1)*2,3,args.img_ht, args.img_wd),
                            nrow=args.seq_len+1, normalize=True, range=(0.0, 3.0))
                        imginfo[mode + '-depths'] = util.to_np(catdepths.unsqueeze(0))

                    if predpts[0] is not None:
                        predpts    = torch.cat(predpts, 0).cpu().float() # (S+1) x 3 x ht x wd
                        gtpts      = pts[id].cpu().float()               # (S+1) x 3 x ht x wd
                        catpts     = torchvision.utils.make_grid(
                            torch.cat([predpts, gtpts],0).view(-1,3,args.img_ht,args.img_wd),
                            nrow=args.seq_len+1, normalize=True, range=(0.0, 3.0))
                        imginfo[mode + '-pts'] = util.to_np(catpts.unsqueeze(0))

                    # Send to tensorboard
                    for tag, images in imginfo.items():
                        tblogger.image_summary(tag, images, iterct)

            # Measure viz time
            viz_time.update(time.time() - start)

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=num_iters, total=num_iters,
                samplecurr=data_loader.niters+1, sampletotal=len(data_loader),
                stats=stats)
    print('========================================================')

    # Return the loss & flow loss
    return stats

### Print statistics
def print_stats(mode, epoch, curr, total, samplecurr, sampletotal,
                stats, bsz=None):
    # Print loss
    bsz = args.batch_size if bsz is None else bsz
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], Batch size: {}, '
          'Loss: {loss.val:.4f} ({loss.avg:.4f}), '.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, bsz, loss=stats.loss))

    # Print flow loss per timestep
    for k in range(args.seq_len+1):
        print('\tStep: {}, Recons: {:.3f} ({:.3f}), '
              'Var-KL: {:.3f} ({:.4f}), '
              'TransEnc-KL => {:.3f} ({:.3f}), '
            .format(
            1 + k * args.step_len,
            stats.reconsloss.val[k], stats.reconsloss.avg[k],
            stats.varklloss.val[k], stats.varklloss.avg[k],
            stats.transencklloss.val[k-1] if (k > 0) else 0,
            stats.transencklloss.avg[k-1] if (k > 0) else 0,
        ))

################ RUN MAIN
if __name__ == '__main__':
    main()
