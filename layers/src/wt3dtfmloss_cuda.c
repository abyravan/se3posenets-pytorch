#include <THC/THC.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include "cuda/wt3dtfmloss_kernel.h"

extern THCState *state;

// =============== FWD PASS ================== //
int Weighted3DTransformLoss_forward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetpoints,
			int sizeAverage)
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
               "Please use NonRigidTransform3D layer + MSE criterion instead \n", nTfmParams);
        assert(false); // Exit
    }

    // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

    // Get data pointers
    float *points_data    = THCudaTensor_data(state, points);
    float *masks_data 	  = THCudaTensor_data(state, masks);
    float *tfms_data      = THCudaTensor_data(state, tfms);
    float *targetpts_data = THCudaTensor_data(state, targetpoints);

	// Get current cuda stream
	cudaStream_t stream = THCState_getCurrentStream(state);

	// Run the kernel
    float loss = Weighted3DTransformLoss_ForwardLauncher(
                      points_data, masks_data, tfms_data, targetpts_data,
                      batchSize, ndim, nrows, ncols, nSE3, nTfmParams,
                      ps, ms, ts,
                      stream);

    // Divide by number of points if asked for average
    loss *= 0.5; // Scale loss by 0.5
    if(sizeAverage)
    {
        long nElements = THCudaTensor_nElement(state, points);
        loss /= ((float)nElements);
    }

    return loss;
}

// =============== BWD PASS ================== //
void Weighted3DTransformLoss_backward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetpoints,
			THCudaTensor *gradPoints,
			THCudaTensor *gradMasks,
			THCudaTensor *gradTfms,
			int sizeAverage)
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
               "Please use NonRigidTransform3D layer + MSE criterion instead \n", nTfmParams);
        assert(false); // Exit
    }

	 // Set gradients w.r.t pts & tfms to zero (as we add to these in a loop later)
    THCudaTensor_fill(state, gradPoints, 0);
    THCudaTensor_fill(state, gradTfms, 0);

    // Get data pointers
    float *points_data        = THCudaTensor_data(state, points);
    float *masks_data         = THCudaTensor_data(state, masks);
    float *tfms_data 		  = THCudaTensor_data(state, tfms);
    float *targetpts_data 	  = THCudaTensor_data(state, targetpoints);
    float *gradPoints_data    = THCudaTensor_data(state, gradPoints);
    float *gradMasks_data 	  = THCudaTensor_data(state, gradMasks);
    float *gradTfms_data      = THCudaTensor_data(state, gradTfms);

	 // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

	// Get current cuda stream
	cudaStream_t stream = THCState_getCurrentStream(state);

	// Run the kernel
    Weighted3DTransformLoss_BackwardLauncher(
		  points_data, masks_data, tfms_data, targetpts_data,
		  gradPoints_data, gradMasks_data, gradTfms_data,
		  batchSize, ndim, nrows, ncols, nSE3, nTfmParams,
		  ps, ms, ts,
		  stream);

    // Average the gradients if sizeaverage is set
    if (sizeAverage)
    {
        // Compute scale factor
        long nElements = THCudaTensor_nElement(state, points);
        float norm = 1.0/((float)nElements);

        // Average gradients
        THCudaTensor_mul(state, gradPoints, gradPoints, norm);
        THCudaTensor_mul(state, gradMasks, gradMasks, norm);
        THCudaTensor_mul(state, gradTfms, gradTfms, norm);
    }
}