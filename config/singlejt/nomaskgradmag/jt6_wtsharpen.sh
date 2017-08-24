#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt6_2se3_wtsharpenr1s0_1seq.yaml
CUDA_VISIBLE_DEVICES=4 python main_ctrlnets.py -c config/singlejt/nomaskgradmag/jt6_4se3_wtsharpenr1s0_1seq.yaml
