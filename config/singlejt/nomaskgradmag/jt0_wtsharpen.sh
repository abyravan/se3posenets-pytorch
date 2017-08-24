#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt0_2se3_wtsharpenr1s0_1seq.yaml
CUDA_VISIBLE_DEVICES=0 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt0_4se3_wtsharpenr1s0_1seq.yaml
