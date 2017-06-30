#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd layers/src/cuda
mkdir lib
echo "Compiling layer kernels by nvcc..."
nvcc -c -o lib/reorg_kernel.cu.o reorg_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
nvcc -c -o lib/roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../
