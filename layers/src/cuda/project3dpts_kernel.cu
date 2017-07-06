#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>

/// Get the (batch,row,col) indices corresponding to a given thread index (3D point index)
__device__ void getCoordinates_1(const int tid, const int nrows, const int ncols,
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

/*
 * Projects points and does the depth test for each of the input points. For each output point:
 * (xout,yout)_i = (xpix,ypix)_i for each input point i
 *  zout_i       = z of the closest point that projects onto it (After projection & depth test)
 */
__global__ void projectPointsAndDepthTest(const float *input_data, float *output_data,
                                          const float fx, const float fy, const float cx, const float cy,
                                          const int batchSize, const int nrows, const int ncols, const int npoints,
                                          const int is0, const int is1, const int is2, const int is3)
{
    // Get the index of the point
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Since they are 1D only
    if (id >= npoints) return;

    // Get the batch, row and column indices
    int b,r,c;
    getCoordinates_1(id, nrows, ncols, b, r, c);

    // Get the 3D input point
    long vali = b*is0 + r*is2 + c*is3; // Don't add stride along 3D dim
    float x = *(input_data + 0*is1 + vali);
    float y = *(input_data + 1*is1 + vali);
    float z = *(input_data + 2*is1 + vali);
    if (z <= 0) return; // No valid projection : Z <= 0

    // Do a perspective transform, scale by focal length & add principal point
    float xpix = ((x/z) * fx) + cx;// + 1; // Points go from [0, row-1] & [0, col-1] in original data
    float ypix = ((y/z) * fy) + cy;// + 1;

    // Check projection success / Check limits / Do the depth test
    float xpixr = round(xpix); // Rounded off pixel col
    float ypixr = round(ypix); // Rounded off pixel row
    if (xpixr >= 0 && xpixr < ncols && ypixr >= 0 && ypixr < nrows)
    {
        // Do depth test:
        //   If z >= z at pixel, discard this point
        //   Else z at pixel = z
        // Note: We use ATOMICMIN here considering the float as an int
        //       This works since our float values are always positive
        // See: https://devtalk.nvidia.com/default/topic/492068/atomicmin-with-float/
        // See: http://stereopsis.com/radix.html
        long valo   = b*is0 + ypixr*is2 + xpixr*is3; // y = row, x = col
        atomicMin((unsigned int*)(output_data + 2*is1 + valo), __float_as_int(z));
        //fatomicMin(output_data + 2*is1 + valo, z);
    }
}

/*
 * Refines the projected points. For each input point, this finds if that point has a valid projection:
 * i.e. if that point is closest to the camera and visible. If so, this point has its index set.
 * If not, that point's values are set to (0,0,0)
 */
__global__ void refineOutput(const float *input_data, float *output_data, float *indexMap_data,
                             const float fx, const float fy, const float cx, const float cy,
                             const int batchSize, const int nrows, const int ncols, const int npoints,
                             const int is0, const int is1, const int is2, const int is3,
                             const int iMs0, const int iMs1, const int iMs2, const int iMs3)
{
    // Get the index of the point
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Since they are 1D only
    if (id >= npoints) return;

    // Get the batch, row and column indices
    int b,r,c;
    getCoordinates_1(id, nrows, ncols, b, r, c);
    long vali = b*is0 + r*is2 + c*is3; // Don't add stride along 3D dim

    // Check the z-value of the output at the present point. If it is HUGE_VAL, set (x,y,z) to zero
    if (*(output_data + 2*is1 + vali) == HUGE_VALF)
    {
        *(output_data + 2*is1 + vali) = 0;
    }

    // Get the 3D input point
    float x = *(input_data + 0*is1 + vali);
    float y = *(input_data + 1*is1 + vali);
    float z = *(input_data + 2*is1 + vali);
    if (z <= 0) return; // No valid projection : Z <= 0

    // Do a perspective transform, scale by focal length & add principal point
    float xpix = ((x/z) * fx) + cx;// + 1; // Points go from [0, row-1] & [0, col-1] in original data
    float ypix = ((y/z) * fy) + cy;// + 1;

    // Check projection success / Check limits / Do the depth test
    float xpixr = round(xpix); // Rounded off pixel col
    float ypixr = round(ypix); // Rounded off pixel row
    if (xpixr >= 0 && xpixr < ncols && ypixr >= 0 && ypixr < nrows)
    {
        // Get the z-value at the pixel corresponding to this input point
        long valo   = b*is0 + ypixr*is2 + xpixr*is3; // y = row, x = col
        float zo    = *(output_data + 2*is1 + valo); // z at output

        // If the z values do not match, this point is not visible. Else:
        // Update the index map (at the output pixel)
        if (zo == z)
        {
            // Set X and Y values to the interpolated pixel values
            *(output_data + 0*is1 + valo) = xpix;
            *(output_data + 1*is1 + valo) = ypix;

            // Set index map value
            long valim  = b*iMs0 + ypixr*iMs2 + xpixr*iMs3; // y = row, x = col
            *(indexMap_data + valim) = vali; // ID of input point for that pixel
        }
    }
}

/*
 * Computes the gradient for the perspective projection + depth test function
 */
__global__ void projectionGradient(const float *input_data, const float *gradOutput_data,
                                   const float *indexMap_data, float *gradInput_data,
                                   const float fx, const float fy,
                                   const int batchSize, const int nrows, const int ncols, const int npoints,
                                   const int is0, const int is1, const int is2, const int is3,
                                   const int iMs0, const int iMs1, const int iMs2, const int iMs3)
{
    // Get the index of the point
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Since they are 1D only
    if (id >= npoints) return;

    // Get the batch, row and column indices
    int b,r,c;
    getCoordinates_1(id, nrows, ncols, b, r, c);

    // Get the index map value (for that output pixel)
    long valim = b*iMs0 + r*iMs2 + c*iMs3; // y = row, x = col
    long vali  = (long)(*(indexMap_data + valim));
    if (vali == -1) return; // In case this point has no corresponding output index, return

    // Get input point (from set of all input points)
    float x = *(input_data + 0*is1 + vali);
    float y = *(input_data + 1*is1 + vali);
    float z = *(input_data + 2*is1 + vali);

    // Get gradOutput value (for that output pixel)
    long valgo = b*is0 + r*is2 + c*is3; // y = row, x = col
    float gx = *(gradOutput_data + 0*is1 + valgo);
    float gy = *(gradOutput_data + 1*is1 + valgo);
    float gz = *(gradOutput_data + 2*is1 + valgo);

    // Gradient w.r.t x = (fx/z) * gx
    // Gradient w.r.t y = (fy/z) * gy
    // Gradient w.r.t z = (-x/z^2) * fx * gx + (-y/z^2) * fy * gy + gz
    *(gradInput_data + 0*is1 + vali) = (fx/z) * gx;
    *(gradInput_data + 1*is1 + vali) = (fy/z) * gy;
    *(gradInput_data + 2*is1 + vali) = ((-x/pow(z,2)) * fx * gx) + ((-y/pow(z,2)) * fy * gy) + gz;
}

// =============== FWD PASS ================== //

int Project3DPointsToSubPixelDepth_ForwardLauncher(const float *input, float *indexMap, float *output,
                                                   int batchSize, int nrows, int ncols,
                                                   float fx, float fy, float cx, float cy,
                                                   const long *is, const long *iMs,
                                                   cudaStream_t stream)
{


    // Block and thread structure - we have one large set of points, so use 1d block/threads
    long npoints       = batchSize * nrows * ncols;
    int numBlocks = ceil(npoints * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    // Project the points and run the depth test first (parallelize across number of points)
    projectPointsAndDepthTest <<< blocks, threads, 0, stream >>>(
        input, output,
        fx, fy, cx, cy,
        batchSize, nrows, ncols, (int)npoints,
        (int) is[0],  (int) is[1],  (int) is[2],  (int)is[3]);

    // Refine the output - only visible points get valid projections. Other points are all zeros.
    refineOutput <<< blocks, threads, 0, stream >>>(
        input, output, indexMap,
        fx, fy, cx, cy,
        batchSize, nrows, ncols, (int)npoints,
        (int) is[0],  (int) is[1],  (int) is[2],  (int)is[3],
        (int) iMs[0], (int) iMs[1], (int) iMs[2], (int)iMs[3]);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in Project3DPointsToSubPixelDepth_ForwardLauncher: %s\n", cudaGetErrorString(err));
        assert(false);
    }

    return 1;
}

// =============== BWD PASS ================== //

int Project3DPointsToSubPixelDepth_BackwardLauncher(const float *input, const float *indexMap,
                                                    float *gradInput, const float *gradOutput,
                                                    int batchSize, int nrows, int ncols,
                                                    float fx, float fy, float cx, float cy,
                                                    const long *is, const long *iMs,
                                                    cudaStream_t stream)
{


    // Block and thread structure - we have one large set of points, so use 1d block/threads
    long npoints       = batchSize * nrows * ncols;
    int numBlocks = ceil(npoints * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    // Run the kernel (parallelize across number of points)
    projectionGradient <<< blocks, threads, 0, stream >>>(
        input, gradOutput, indexMap, gradInput,
        fx, fy,
        batchSize, nrows, ncols, (int)npoints,
        (int) is[0],  (int) is[1],  (int) is[2],  (int)is[3],
        (int) iMs[0], (int) iMs[1], (int) iMs[2], (int)iMs[3]);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in Project3DPointsToSubPixelDepth_BackwardLauncher: %s\n", cudaGetErrorString(err));
        assert(false);
    }

    return 1;
}

#ifdef __cplusplus
}
#endif