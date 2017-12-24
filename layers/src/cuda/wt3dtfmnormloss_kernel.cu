#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define WARPSIZE 32

__constant__ float constTfms[15000];  // ... or some other big enough number

// Warp-shuffle to compute the sum across the warp very efficiently
__inline__ __device__
float warpReduceSum_1(float val) {
  for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

/// Get the (batch,row,col) indices corresponding to a given thread index (3D point index)
__device__ void getCoordinates_4(const int tid, const int nrows, const int ncols,
                                 int &batch, int &row, int &col)
{
    // Get col id
    int id = tid;
    col = id % ncols;
    id = id / ncols;

    // Get row id
    row = id % nrows;
    id = id / nrows;

    // Get batch id
    batch = id;
}

// =============== FWD PASS ================== //

///////////// Kernel
// Compute the loss by transforming each input point by all the "k" transforms, measuring the error
// between the prediction and the target and weighing the corresponding error by the predicted mask weight
__global__ void computeLoss_f(const float *inputpts, const float *masks, const float *targetflows, const float *numpts,
                              float *devLoss, int nrows, int ncols, int npoints, int nSE3,
                              float normWt, int normPerPt,
                              int ps0, int ps1, int ps2, int ps3,
                              int ms0, int ms1, int ms2, int ms3,
                              int ts0, int ts1, int ts2, int ts3)
{
    // Get the index of the point
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Since they are 1D only

    // Create a shared memory buffer for storing the gradients w.r.t a single transform
    extern __shared__ float sharedLoss[];

    // Declare temp vars
    int tid = threadIdx.x; // Id of thread in local block
    int nThreads = blockDim.x;

    // Compute loss only if the point is within limits
    sharedLoss[tid] = 0; // Initialize to zero
    if (id < npoints)
    {
        // Get the batch, row and column indices
        int b,r,c;
        getCoordinates_4(id, nrows, ncols, b, r, c);

        // Get 3D input point (p)
        int valp = b*ps0 + r*ps2 + c*ps3; // Don't add stride along 3D dim
        float x = *(inputpts + 0*ps1 + valp);
        float y = *(inputpts + 1*ps1 + valp);
        float z = *(inputpts + 2*ps1 + valp);

        // Get 3D target point (pt)
        float fxt = *(targetflows + 0*ps1 + valp);
        float fyt = *(targetflows + 1*ps1 + valp);
        float fzt = *(targetflows + 2*ps1 + valp);

        // Get normalizing constant (sigma), clamped to a min of 2e-3
        float sx = 0, sy = 0, sz = 0, s = 0;
        if (normPerPt)
        {
            s = fmaxf(normWt * powf(fxt*fxt + fyt*fyt + fzt*fzt, 0.5), 2e-3); // Scale by length of flow vector
        }
        else
        {
            // Independent per dimension of the flow vector
            sx = fmaxf(normWt * fabsf(fxt), 2e-3);
            sy = fmaxf(normWt * fabsf(fyt), 2e-3);
            sz = fmaxf(normWt * fabsf(fzt), 2e-3);
        }

        // Compute sum_k w_k * ||R_k*p + t_k - pt||^2 across the different SE3s
        int valm = b*ms0 + r*ms2 + c*ms3;
        for (int k = 0; k < nSE3; k++)
        {
            // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
            float *T = constTfms + b*ts0 + k*ts1;   // Get the 'k'th transform
            float xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
            float yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
            float zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

            // Compute flow error (predicted - target flow)
            float ex = (xp - x) - fxt;
            float ey = (yp - y) - fyt;
            float ez = (zp - z) - fzt;

            // Compute normalized error (different scalar per dimension)
            float err;
            if (normPerPt)
                err = (ex*ex + ey*ey + ez*ez) / s;
            else
                err = (ex*ex)/sx + (ey*ey)/sy + (ez*ez)/sz; // different scale per dimension

            // Weight the error by the mask weight
            float w_k = *(masks + k*ms1 + valm); // Get the weight for the 'k'th component of the error

            // Store scaled loss in shared memory
            sharedLoss[tid] += w_k * err / numpts[b];
        }
    }
    __syncthreads();

    // === Do the parallel reduce for that particular transform dimension
    // === ASSUMPTION: We have power of 2 block sizes!
    // From: Slide 22 of http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    for(unsigned int s = nThreads/2; s>=32; s>>=1)
    {
        // Second nThreads/2 elements will be added to first nThreads/2 elements, then
        // Second nThreads/4 elements will be added to first nThreads/4 elements and so on!
        if (tid < s)
            sharedLoss[tid] += sharedLoss[tid + s];
        __syncthreads();
    }

    // This uses warp-shuffle to compute the sum across a warp (32 threads)
    // Note that for this to work, you have to have run the loop until the sum is computed for the first 32 threads in the warp
    if (tid < 32)
    {
        float sum = warpReduceSum_1(sharedLoss[tid]);
        if (tid == 0)
	    atomicAdd(devLoss, (float)sum);
    }
}

///////////////// FWD pass launcher
int Weighted3DTransformNormLoss_ForwardLauncher(const float *points, const float *masks, const float *tfms, const float *targetflows, const float *numpts,
								  int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
                                  float normWt, int normPerPt,
								  const long *ps, const long *ms, const long *ts,
								  cudaStream_t stream)
{
    // Copy transforms to constant memory to reduce global memory read overhead
    cudaMemcpyToSymbol(constTfms, tfms, nTfmParams * sizeof(float));

    // Block and thread structure - we have one large set of points, so use 1d block/threads
    int npoints = batchSize * nrows * ncols;
    int numBlocks = ceil(npoints * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    // Allocate memory for loss on gpu
    float loss;
    float *devloss;
    cudaMalloc((void**)&devloss, sizeof(float));
    cudaMemset(devloss, 0, sizeof(float));

//    // Timer
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    // Project the points and run the depth test first (parallelize across number of points)
    computeLoss_f <<< blocks, threads, 256*sizeof(float), stream >>>(
                                                                     points,
                                                                     masks,
                                                                     targetflows,
                                                                     numpts,
                                                                     devloss,
                                                                     nrows,
                                                                     ncols,
                                                                     npoints,
                                                                     nSE3,
                                                                     normWt,
                                                                     normPerPt,
                                                                     (int) ps[0],
                                                                     (int) ps[1],
                                                                     (int) ps[2],
                                                                     (int) ps[3],
                                                                     (int) ms[0],
                                                                     (int) ms[1],
                                                                     (int) ms[2],
                                                                     (int) ms[3],
                                                                     (int) ts[0],
                                                                     (int) ts[1],
                                                                     (int) ts[2],
                                                                     (int) ts[3]
                                                                                 );

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy over the loss value
    cudaMemcpy(&loss, devloss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devloss); // Free memory

//    // Finish timing and show stats
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("FWD: Time taken in milliseconds: %f\n",milliseconds);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in Weighted3DTransformNormLoss_ForwardLauncher: %s\n", cudaGetErrorString(err));
        assert(false);
    }

    return loss;
}

// ============= BWD PASS =================== //

// Compute the gradients w.r.t input points & masks given gradients w.r.t output 3D points
__global__ void computeLossGradients_f(const float *inputpts, const float *masks, const float *numpts,
                                       float *gradInputpts, float *gradMasks, float *gradTfms,
                                       const float *targetflows,
                                       int nrows, int ncols, int nSE3,
                                       float normWt, int normPerPt,
                                       int ps0, int ps1, int ps2, int ps3,
                                       int ms0, int ms1, int ms2, int ms3,
                                       int ts0, int ts1, int ts2, int ts3)
{
    // Get the row, col, batch IDs & figure out if we are within limits
    int c = (blockIdx.x * blockDim.x) + threadIdx.x; // col ID (innermost dimension in our data for coalescing)
    int r = (blockIdx.y * blockDim.y) + threadIdx.y; // row ID
    int b = blockIdx.z; // Batch ID (since blockDim.z = 1, theadIdx.z = 0)
    bool withinLimits = ((c < ncols) && (r < nrows));

    // Create a shared memory buffer for storing the gradients w.r.t a single transform
    extern __shared__ float sharedData[];

    // Declare temp vars
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // Id of thread in local block
    int nThreads = blockDim.x * blockDim.y;
    int nThreads2 = nThreads/2;
    int nSharedGrads  = nThreads * 12;
    int nSharedGradResults = nSE3*12;
    float *sharedGradTfms = sharedData; // nThreads*12
    float *sharedGradTfmResults = (float *)&sharedData[nSharedGrads]; // nSE3*12

    // Get 3D input point (p) & target flow (gpt). Read only if inside limits
    float x, y, z, fxt, fyt, fzt;
    float sx = 0, sy = 0, sz = 0, s = 0;
    int valp = b*ps0 + r*ps2 + c*ps3; // Don't add stride along 3D dim
    if (withinLimits)
    {
        x = *(inputpts + 0*ps1 + valp);
        y = *(inputpts + 1*ps1 + valp);
        z = *(inputpts + 2*ps1 + valp);

        // Get gradient w.r.t output point (gpt)
        fxt = *(targetflows + 0*ps1 + valp);
        fyt = *(targetflows + 1*ps1 + valp);
        fzt = *(targetflows + 2*ps1 + valp);

        // Get normalizing constant (sigma), clamped to a min of 2e-3
        if (normPerPt)
        {
            s = fmaxf(normWt * powf(fxt*fxt + fyt*fyt + fzt*fzt, 0.5), 2e-3); // Scale by length of flow vector
        }
        else
        {
            // Independent per dimension of the flow vector
            sx = fmaxf(normWt * fabsf(fxt), 2e-3);
            sy = fmaxf(normWt * fabsf(fyt), 2e-3);
            sz = fmaxf(normWt * fabsf(fzt), 2e-3);
        }
    }

    // Compute the gradients over all the transforms from a given 3D point
    int valm = b*ms0 + r*ms2 + c*ms3;
    float gx = 0, gy = 0, gz = 0; // Grads w.r.t input pts
    for(int k = 0; k < nSE3; k++)
    {
        // Compute all the gradients if within limits or set the grads to zero
        if(withinLimits)
        {
            // Get transform & wt
            float w_k = *(masks + k*ms1 + valm);   // Get the weight for the 'k'th transform "
            float *T  = constTfms + b*ts0 + k*ts1; // Get the 'k'th transform

            // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
            float xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
            float yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
            float zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

            // Compute flow error (predicted - target flow)
            float ex = (xp - x) - fxt;
            float ey = (yp - y) - fyt;
            float ez = (zp - z) - fzt;

            // === Gradient w.r.t mask (w_k) = 0.5 * err
            float err;
            if (normPerPt)
                err = (ex*ex + ey*ey + ez*ez) / s;
            else
                err = (ex*ex)/sx + (ey*ey)/sy + (ez*ez)/sz; // different scale per dimension
            *(gradMasks + k*ms1 + valm) = 0.5*err / numpts[b];

            // == Scale error terms by sigma (from here on we only use the scaled terms)
            ex /= normPerPt ? s : sx;
            ey /= normPerPt ? s : sy;
            ez /= normPerPt ? s : sz;

            // === Gradient w.r.t input points (p)
            // (p = w_k * (R^T - I) * diff/sigma, summed across all the "k" transforms)
            gx += w_k * ((T[0]-1.0) * ex + T[4]       * ey + T[8]        * ez);
            gy += w_k * (T[1]       * ex + (T[5]-1.0) * ey + T[9]        * ez);
            gz += w_k * (T[2]       * ex + T[6]       * ey + (T[10]-1.0) * ez);

            // === Gradients w.r.t transforms (t_k), stored in shared memory
            // Divide by numpts
            ex /= numpts[b];
            ey /= numpts[b];
            ez /= numpts[b];

            // Grads w.r.t rotation parameters (sum across all pts)
            // First nThreads params is Tfm(0,0), next is Tfm(0,1) etc for removing memory bank conflicts when reading to shared memory
            sharedGradTfms[0*nThreads+tid]  = w_k * x * ex;
            sharedGradTfms[1*nThreads+tid]  = w_k * y * ex;
            sharedGradTfms[2*nThreads+tid]  = w_k * z * ex;
            sharedGradTfms[4*nThreads+tid]  = w_k * x * ey;
            sharedGradTfms[5*nThreads+tid]  = w_k * y * ey;
            sharedGradTfms[6*nThreads+tid]  = w_k * z * ey;
            sharedGradTfms[8*nThreads+tid]  = w_k * x * ez;
            sharedGradTfms[9*nThreads+tid]  = w_k * y * ez;
            sharedGradTfms[10*nThreads+tid] = w_k * z * ez;

            // Grads w.r.t translation parameters (sum across all pts)
            sharedGradTfms[3*nThreads+tid]  = w_k * ex;
            sharedGradTfms[7*nThreads+tid]  = w_k * ey;
            sharedGradTfms[11*nThreads+tid] = w_k * ez;
        }
        else
        {
            // Re-initialize shared memory to zero (no need to sync here as we don't += to this memory till we do a syncthreads later)
            for(int i = tid; i < nSharedGrads; i+=nThreads)
                sharedGradTfms[i] = 0;
        }
        __syncthreads(); // Synchronize all threads before we sum up the tfm gradients

        // === Do the parallel reduce for that particular transform dimension
        // === ASSUMPTION: We have power of 2 block sizes!
        // From: Slide 22 of http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
        // We use first half of threads to compute sums for first 6 transform params & the rest for the last 6 params
        for(unsigned int s = nThreads2; s>=32; s>>=1)
        {
            // Second nThreads/2 elements will be added to first nThreads/2 elements, then
            // Second nThreads/4 elements will be added to first nThreads/4 elements and so on!
            if (tid < s)
            {
                // Sum up gradients w.r.t first 6 parameters!
                for(int i = 0; i < 6; i++)
                    sharedGradTfms[i*nThreads + tid] += sharedGradTfms[i*nThreads + tid + s];
            }
            else if((tid >= nThreads2) && (tid - nThreads2) < s) // Use the second half of threads to process the remaining 6 transform parameters
            {
                // Sum up gradients w.r.t last 6 parameters!
                for(int i = 6; i < 12; i++)
                    sharedGradTfms[i*nThreads + tid - nThreads2] += sharedGradTfms[i*nThreads + tid - nThreads2 + s];
            }
            __syncthreads();
        }

        // This uses warp-shuffle to compute the sum across a warp (32 threads)
        // Note that for this to work, you have to have run the loop until the sum is computed for the first 32 threads in the warp
        if (tid < 32)
        {
            for(int i = 0; i < 12; i++)
            {
                float sum = warpReduceSum_1(sharedGradTfms[i*nThreads + tid]); // Declared elsewhere
                if (tid == 0)
                    sharedGradTfmResults[k*12+i] = sum; // Store final summed result in shared memory, we can copy to global later in parallel
            }
        }
    }
    __syncthreads(); // Wait till all gradients have been propely summed up!

    // Add computed tfm gradients to global memory in parallel!
    for(int i = tid; i < nSharedGradResults; i+=nThreads)
        atomicAdd(gradTfms + b*ts0 + i, sharedGradTfmResults[i]); // Final value corresponding to that term of the tfm

    // Gradients w.r.t pts (copy after sum across tfms)
    if (withinLimits)
    {
        *(gradInputpts + 0*ps1 + valp) = gx / numpts[b];
        *(gradInputpts + 1*ps1 + valp) = gy / numpts[b];
        *(gradInputpts + 2*ps1 + valp) = gz / numpts[b];
    }
}

////////////////////////////////////
// == BWD pass code
void Weighted3DTransformNormLoss_BackwardLauncher(const float *points, const float *masks, const float *tfms, const float *targetflows, const float *numpts,
                                              float *gradPoints, float *gradMasks, float *gradTfms,
                                              int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
                                              float normWt, int normPerPt,
                                              const long *ps, const long *ms, const long *ts,
                                              cudaStream_t stream)
{
    // Copy transforms to constant memory to reduce global memory read overhead
    cudaMemcpyToSymbol(constTfms, tfms, nTfmParams * sizeof(float));

    // Compute gradients w.r.t the input tfms next
    dim3 threads(16,16,1);
    dim3 blocks(ceil(ncols*(1.0/threads.x)),ceil(nrows*(1.0/threads.y)),batchSize); // all threads in a block will access same example
    int sharedMemSize = threads.x * threads.y * 3 * 4 * sizeof(float) + nSE3 * 3 * 4 * sizeof(float); // Memory for 12 vals per thread + nSE3*12 vals for storing result
    if (sharedMemSize > 32000)
    {
        printf("Shared memory size for transform gradients (%d) > 32000. Can't be stored in shared memory."
               "Please use NonRigidTransform3D layer + MSE criterion or reduce number of threads per block \n", sharedMemSize);
        assert(false); // Exit
    }

//    // Timer
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    computeLossGradients_f<<< blocks, threads, sharedMemSize, stream >>>(
                                                                        points,
                                                                        masks,
                                                                        numpts,
                                                                        gradPoints,
                                                                        gradMasks,
                                                                        gradTfms,
                                                                        targetflows,
                                                                        nrows,
                                                                        ncols,
                                                                        nSE3,
                                                                        normWt,
                                                                        normPerPt,
                                                                        (int) ps[0],
                                                                        (int) ps[1],
                                                                        (int) ps[2],
                                                                        (int) ps[3],
                                                                        (int) ms[0],
                                                                        (int) ms[1],
                                                                        (int) ms[2],
                                                                        (int) ms[3],
                                                                        (int) ts[0],
                                                                        (int) ts[1],
                                                                        (int) ts[2],
                                                                        (int) ts[3]
                                                                                    );

    // Wait for kernel to finish
    cudaDeviceSynchronize();

//    // Finish timing and show stats
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("BWD: Time taken in milliseconds: %f\n",milliseconds);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in Weighted3DTransformNormLoss_BackwardLauncher: %s\n", cudaGetErrorString(err));
        assert(false);
    }
}

#ifdef __cplusplus
}
#endif
