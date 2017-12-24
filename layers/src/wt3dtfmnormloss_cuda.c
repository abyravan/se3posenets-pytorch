#include <THC/THC.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include "cuda/wt3dtfmnormloss_kernel.h"

extern THCState *state;

// =============== FWD PASS ================== //
int Weighted3DTransformNormLoss_forward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetflows,
            THCudaTensor *numpts,
			float normWt,
			int normPerPt)
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

    // New memory in case the inputs are not contiguous
    points = THCudaTensor_newContiguous(state, points);
    masks  = THCudaTensor_newContiguous(state, masks);
    tfms   = THCudaTensor_newContiguous(state, tfms);
    targetflows = THCudaTensor_newContiguous(state, targetflows);
    numpts = THCudaTensor_newContiguous(state, numpts);

    // Get data pointers
    float *points_data      = THCudaTensor_data(state, points);
    float *masks_data 	    = THCudaTensor_data(state, masks);
    float *tfms_data        = THCudaTensor_data(state, tfms);
    float *targetflows_data = THCudaTensor_data(state, targetflows);
    float *numpts_data      = THCudaTensor_data(state, numpts);

    // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

	// Get current cuda stream
	cudaStream_t stream = THCState_getCurrentStream(state);

	// Run the kernel
    float loss = Weighted3DTransformNormLoss_ForwardLauncher(
                      points_data, masks_data, tfms_data, targetflows_data, numpts_data,
                      batchSize, ndim, nrows, ncols, nSE3, nTfmParams,
                      normWt, normPerPt,
                      ps, ms, ts,
                      stream);

    // Divide by number of points if asked for average
    loss /= (2.0 * ((double) batchSize));

    // Free memory
    THCudaTensor_free(state, points);
    THCudaTensor_free(state, masks);
    THCudaTensor_free(state, tfms);
    THCudaTensor_free(state, targetflows);
    THCudaTensor_free(state, numpts);

    return loss;
}

// =============== BWD PASS ================== //
void Weighted3DTransformNormLoss_backward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetflows,
            THCudaTensor *numpts,
			THCudaTensor *gradPoints,
			THCudaTensor *gradMasks,
			THCudaTensor *gradTfms,
            THCudaTensor *gradOutput,
            float normWt,
            int normPerPt)
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

    // New memory in case the inputs are not contiguous
    points = THCudaTensor_newContiguous(state, points);
    masks  = THCudaTensor_newContiguous(state, masks);
    tfms   = THCudaTensor_newContiguous(state, tfms);
    targetflows = THCudaTensor_newContiguous(state, targetflows);
    numpts = THCudaTensor_newContiguous(state, numpts);

    // Get data pointers
    float *points_data        = THCudaTensor_data(state, points);
    float *masks_data         = THCudaTensor_data(state, masks);
    float *tfms_data 		  = THCudaTensor_data(state, tfms);
    float *targetflows_data   = THCudaTensor_data(state, targetflows);
    float *gradPoints_data    = THCudaTensor_data(state, gradPoints);
    float *gradMasks_data 	  = THCudaTensor_data(state, gradMasks);
    float *gradTfms_data      = THCudaTensor_data(state, gradTfms);
    float *gradOutput_data    = THCudaTensor_data(state, gradOutput);
    float *numpts_data 		  = THCudaTensor_data(state, numpts);

	 // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

	// Get current cuda stream
	cudaStream_t stream = THCState_getCurrentStream(state);

	// Run the kernel
    Weighted3DTransformNormLoss_BackwardLauncher(
		  points_data, masks_data, tfms_data, targetflows_data, numpts_data,
		  gradPoints_data, gradMasks_data, gradTfms_data,
		  batchSize, ndim, nrows, ncols, nSE3, nTfmParams,
		  normWt, normPerPt,
		  ps, ms, ts,
		  stream);

    // Get gradient w.r.t output
    float norm;
    cudaMemcpy(&norm, gradOutput_data, sizeof(float), cudaMemcpyDeviceToHost);
    norm /= (float) batchSize;

    // Scale by grad output & average gradients
    THCudaTensor_mul(state, gradPoints, gradPoints, norm);
    THCudaTensor_mul(state, gradMasks, gradMasks, norm);
    THCudaTensor_mul(state, gradTfms, gradTfms, norm);

    // Free memory
    THCudaTensor_free(state, points);
    THCudaTensor_free(state, masks);
    THCudaTensor_free(state, tfms);
    THCudaTensor_free(state, targetflows);
    THCudaTensor_free(state, numpts);
}
