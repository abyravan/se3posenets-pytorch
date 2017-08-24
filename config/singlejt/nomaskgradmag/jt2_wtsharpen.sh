#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt2_2se3_wtsharpenr1s0_1seq.yaml
CUDA_VISIBLE_DEVICES=1 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt2_4se3_wtsharpenr1s0_1seq.yaml
