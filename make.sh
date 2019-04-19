#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

if [ -d "$CUDA_PATH" ]; then
	cd layers/src/cuda
	mkdir lib
	echo "Compiling CUDA kernels using nvcc..."
	nvcc -c -o lib/ntfm3d_kernel.cu.o ntfm3d_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
	nvcc -c -o lib/project3dpts_kernel.cu.o project3dpts_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
	cd ../../
	python build.py
	cd ../
else
	echo "Cannot find CUDA installation, just building the CPU version of the layers"
	cd layers
	python build.py
	cd ../
fi
