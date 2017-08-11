#include <THC/THC.h>
#include <math.h>
#include <assert.h>
#include "cuda/project3dpts_kernel.h"

extern THCState *state;

// =============== FWD PASS ================== //
int Project3DPointsToSubPixelDepth_forward_cuda(
                                                THCudaTensor *input,
                                                THCudaTensor *indexMap,
                                                THCudaTensor *output,
                                                float fy, float fx,
                                                float cy, float cx)
{
    // Initialize vars
    int batchSize      = input->size[0];
    int ndim           = input->size[1];
    int nrows          = input->size[2];
    int ncols          = input->size[3];
    THAssert(ndim == 3); // 3D points

    // Fill with defaults
    THCudaTensor_fill(state, output, 0);
    THCudaTensor_fill(state, indexMap, -1); // -1 means invalid

    // Fill "z" with HUGE_VALF so that atomicMin works nicely
    THCudaTensor *temp = THCudaTensor_new(state);
    THCudaTensor_narrow(state, temp, output, 1, 2, 1); // Choose "z" value
    THCudaTensor_fill(state, temp, HUGE_VALF); // Set z = HUGE_VALF by default

    // Get strides
    long *is    = input->stride;
    long *iMs   = indexMap->stride;

    // New memory in case the inputs are not contiguous
    input = THCudaTensor_newContiguous(state, input);

    // Get data pointers
    float *input_data 		= THCudaTensor_data(state, input);
    float *indexMap_data 	= THCudaTensor_data(state, indexMap);
    float *output_data 		= THCudaTensor_data(state, output);

    // Get current cuda stream
    cudaStream_t stream = THCState_getCurrentStream(state);

    // Run the kernel
    Project3DPointsToSubPixelDepth_ForwardLauncher(
		  input_data, indexMap_data, output_data,
		  batchSize, nrows, ncols,
		  fx, fy, cx, cy,
		  is, iMs,
		  stream);

    // Free memory
    THCudaTensor_free(state, input);

    return 1;
}

// =============== BWD PASS ================== //
int Project3DPointsToSubPixelDepth_backward_cuda(
                                                 THCudaTensor *input,
                                                 THCudaTensor *indexMap,
                                                 THCudaTensor *gradInput,
                                                 THCudaTensor *gradOutput,
                                                 float fy, float fx,
                                                 float cy, float cx)
{
    // Initialize vars
    int batchSize      = input->size[0];
    int ndim           = input->size[1];
    int nrows          = input->size[2];
    int ncols          = input->size[3];
    THAssert(ndim == 3); // 3D points

    // Fill with defaults
    THCudaTensor_fill(state, gradInput, 0);

    // New memory in case the inputs are not contiguous
    input = THCudaTensor_newContiguous(state, input);

    // Get data pointers
    float *input_data 		= THCudaTensor_data(state, input);
    float *gradOutput_data 	= THCudaTensor_data(state, gradOutput);
    float *indexMap_data 	= THCudaTensor_data(state, indexMap);
    float *gradInput_data 	= THCudaTensor_data(state, gradInput);

    // Get strides
    long *is    = input->stride;
    long *iMs   = indexMap->stride;

    // Get current cuda stream
    cudaStream_t stream = THCState_getCurrentStream(state);

    // Run the kernel
    Project3DPointsToSubPixelDepth_BackwardLauncher(
		  input_data, indexMap_data, gradInput_data, gradOutput_data,
		  batchSize, nrows, ncols,
		  fx, fy, cx, cy,
		  is, iMs,
		  stream);

    // Free memory
    THCudaTensor_free(state, input);

    return 1;
}