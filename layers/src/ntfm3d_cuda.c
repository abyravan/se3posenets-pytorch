#include <THC/THC.h>
#include <math.h>
#include <assert.h>
#include "cuda/ntfm3d_kernel.h"

extern THCState *state;

// =============== FWD PASS ================== //
int NTfm3D_forward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *tfmpoints)
{
    // Initialize vars
    int batchSize = points->size[0];
    int ndim      = points->size[1];
    int nrows     = points->size[2];
    int ncols     = points->size[3];
    int nSE3      = masks->size[1];
    assert(ndim == 3); // 3D points

	 // Check if we can fit all transforms within constant memory
    int nTfmParams = THCudaTensor_nElement(state, tfms);
    if (nTfmParams > 15000)
    {
        printf("Number of transform parameters (%d) > 15000. Can't be stored in constant memory."
               "Please reduce batch size or number of SE3 transforms. \n", nTfmParams);
        assert(false); // Exit
    }

    // Resize output
    THCudaTensor_resizeAs(state, tfmpoints, points);

	 // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

    // Get data pointers
    float *points_data 	  = THCudaTensor_data(state, points);
    float *masks_data 	  = THCudaTensor_data(state, masks);
    float *tfms_data      = THCudaTensor_data(state, tfms);
    float *tfmpoints_data = THCudaTensor_data(state, tfmpoints);

	 // Get current cuda stream
	 cudaStream_t stream = THCState_getCurrentStream(state);

	 // Run the kernel
    NTfm3DForwardLauncher(
		  points_data, masks_data, tfms_data, tfmpoints_data,
		  batchSize, ndim, nrows, ncols, nSE3, nTfmParams,
		  ps, ms, ts,
		  stream);

    return 1;
}

// =============== BWD PASS ================== //
int NTfm3D_backward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *tfmpoints,
			THCudaTensor *gradPoints,
			THCudaTensor *gradMasks,
			THCudaTensor *gradTfms,
			THCudaTensor *gradTfmpoints)
{
    // Initialize vars
    long batchSize = points->size[0];
    long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];
    assert(ndim == 3); // 3D points

    // Check if we can fit all transforms within constant memory
    int nTfmParams = THCudaTensor_nElement(state, tfms);
    if (nTfmParams > 15000)
    {
        printf("Number of transform parameters (%d) > 15000. Can't be stored in constant memory."
               "Please use NonRigidTransform3D layer instead \n", nTfmParams);
        assert(false); // Exit
    }

	 // Set gradients w.r.t pts & tfms to zero (as we add to these in a loop later)
    THCudaTensor_fill(state, gradPoints, 0);
    THCudaTensor_fill(state, gradTfms, 0);

    // Get data pointers
    float *points_data        = THCudaTensor_data(state, points);
    float *masks_data         = THCudaTensor_data(state, masks);
    float *tfms_data 			= THCudaTensor_data(state, tfms);
    float *tfmpoints_data 		= THCudaTensor_data(state, tfmpoints);
    float *gradPoints_data 	= THCudaTensor_data(state, gradPoints);
    float *gradMasks_data 	   = THCudaTensor_data(state, gradMasks);
    float *gradTfms_data      = THCudaTensor_data(state, gradTfms);
    float *gradTfmpoints_data = THCudaTensor_data(state, gradTfmpoints);

	 // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

	 // Get current cuda stream
	 cudaStream_t stream = THCState_getCurrentStream(state);

	 // Run the kernel
    NTfm3DBackwardLauncher(
		  points_data, masks_data, tfms_data, tfmpoints_data,
		  gradPoints_data, gradMasks_data, gradTfms_data, gradTfmpoints_data,
		  batchSize, ndim, nrows, ncols, nSE3, nTfmParams,
		  ps, ms, ts,
		  stream);

    return 1;
}
