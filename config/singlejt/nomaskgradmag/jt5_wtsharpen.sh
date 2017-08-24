#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt5_2se3_wtsharpenr1s0_1seq.yaml
CUDA_VISIBLE_DEVICES=3 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt5_4se3_wtsharpenr1s0_1seq.yaml
