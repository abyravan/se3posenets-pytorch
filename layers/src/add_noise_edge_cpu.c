#include <TH/TH.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Random double from 0 - 1.0
float randFloat()
{
    return ((float) rand() / RAND_MAX);
}

float randNormal()
 /* normal distribution, centered on 0, std dev 1 */
{
  return sqrt(-2*log(randFloat())) * cos(2*M_PI*randFloat());
}

// ===== FLOAT DATA

int AddNoise_float(
            THFloatTensor *depth,
            THFloatTensor *depth_n,
            float zthresh,
            float edgeprob,
            float defprob,
            float noisestd)
{
    // Initialize vars
    long batchsize = depth->size[0];
    long ndim      = depth->size[1];
    long nrows     = depth->size[2];
    long ncols     = depth->size[3];
    assert(ndim == 3);

    // New memory in case the inputs are not contiguous (no need for the local stuff since its temp memory)
    depth = THFloatTensor_newContiguous(depth);
    depth_n = THFloatTensor_newContiguous(depth_n);

    // Get data pointers
    const float *depth_data = THFloatTensor_data(depth);
    float *depthn_data = THFloatTensor_data(depth_n);

    // Get strides
    long *ds = depth->stride;

    /// ====== Iterate over all points, compute local coordinates
    long b,r,c;
    for(b = 0; b < batchsize; b++)
    {
        for(r = 1; r < nrows-1; r++)
        {
            for(c = 1; c < ncols-1; c++)
            {
                /// === Check central diff for large gradients in x & y
                // Get cam pt @ t
                float zr = *(depth_data + b*ds[0] + r*ds[2] + (c+1)*ds[3]); // r, c+1
                float zl = *(depth_data + b*ds[0] + r*ds[2] + (c-1)*ds[3]); // r, c-1
                float zu = *(depth_data + b*ds[0] + (r-1)*ds[2] + c*ds[3]); // r-1, c
                float zd = *(depth_data + b*ds[0] + (r+1)*ds[2] + c*ds[3]); // r+1, c
                float zdiff = (fabsf(zl - zr) > zthresh) + (fabsf(zu - zd) > zthresh); // Edge in x & y

                // Check if edge, if so based on edge prob this pixel is set to zero (not visible)
                float eprob = zdiff * edgeprob;
                if (randFloat() < eprob){
                    *(depthn_data + b*ds[0] + r*ds[2] + c*ds[3]) = 0; // Zero out depth
                    continue;
                }

                // Else, use smaller def prob to set pixels to zero at random
                if (randFloat() < defprob)
                {
                    *(depthn_data + b*ds[0] + r*ds[2] + c*ds[3]) = 0; // Zero out depth
                    continue;
                }

                // Else add noise to pixel based on provided std. deviation
                if (*(depth_data + b*ds[0] + r*ds[2] + c*ds[3]) == 0)
                    continue; //BG has no noise

                // Otherwise add small Gaussian noise
                float znoise = randNormal() * noisestd;
                *(depthn_data + b*ds[0] + r*ds[2] + c*ds[3]) += znoise; // Add noise to depth
            }
        }
    }

    /// ========= Free created memory
    THFloatTensor_free(depth);
    THFloatTensor_free(depth_n);

    // Return
    return 1;
}

