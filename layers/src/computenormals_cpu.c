#include <TH/TH.h>
#include <assert.h>
#include <math.h>

// ===== FLOAT DATA

int ComputeNormals_float(
            THFloatTensor *cloud_1,
            THFloatTensor *cloud_2,
            THByteTensor  *label_1,
            THFloatTensor *deltaposes_12,
            THFloatTensor *normals_1,
            THFloatTensor *tnormals_2,
            float maxdepthdiff)
{
    // Initialize vars
    long batchsize = cloud_1->size[0];
    long ndim      = cloud_1->size[1];
    long nrows     = cloud_1->size[2];
    long ncols     = cloud_1->size[3];
    assert(ndim == 3);

    // New memory in case the inputs are not contiguous
    cloud_1       = THFloatTensor_newContiguous(cloud_1);
    cloud_2       = THFloatTensor_newContiguous(cloud_2);
    label_1       = THByteTensor_newContiguous(label_1);
    deltaposes_12 = THFloatTensor_newContiguous(deltaposes_12);
    normals_1     = THFloatTensor_newContiguous(normals_1);
    tnormals_2    = THFloatTensor_newContiguous(tnormals_2);

    // Get data pointers
    const float *cloud1_data 	     = THFloatTensor_data(cloud_1);
    const float *cloud2_data 	     = THFloatTensor_data(cloud_2);
    const unsigned char *label1_data = THByteTensor_data(label_1);
    const float *deltaposes12_data   = THFloatTensor_data(deltaposes_12);
    float *normals1_data             = THFloatTensor_data(normals_1);
    float *tnormals2_data            = THFloatTensor_data(tnormals_2);

    // Set visibility to zero by default
    THFloatTensor_fill(normals_1, 0);
    THFloatTensor_fill(tnormals_2, 0);

    // Get strides
    long *cs = cloud_1->stride;
    long *ls = label_1->stride;
    long *ps = deltaposes_12->stride;

    /// ====== Iterate over all points, compute normals, rotate them
    long b,r,c;
    for(b = 0; b < batchsize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                /// === Compute normals
                // Get center point
                long valc = b*cs[0] + r*cs[2] + c*cs[3]; // Don't add stride along 3D dim
                float xc = *(cloud1_data + 0*cs[1] + valc);
                float yc = *(cloud1_data + 1*cs[1] + valc);
                float zc = *(cloud1_data + 2*cs[1] + valc);

                // Read data from horizontal neighbours (on same row)
                float hSign = 1;
                float xh = 0, yh = 0, zh = 0;
                if (c-1 >= 0)
                {
                    long valh = b*cs[0] + r*cs[2] + (c-1)*cs[3]; // Don't add stride along 3D dim
                    xh = *(cloud1_data + 0*cs[1] + valh);
                    yh = *(cloud1_data + 1*cs[1] + valh);
                    zh = *(cloud1_data + 2*cs[1] + valh);
                }
                if ((zh == 0) && (c+1 < ncols))// if r-1 is out of limits this will trigger
                {
                    long valh = b*cs[0] + r*cs[2] + (c+1)*cs[3]; // Don't add stride along 3D dim
                    xh = *(cloud1_data + 0*cs[1] + valh);
                    yh = *(cloud1_data + 1*cs[1] + valh);
                    zh = *(cloud1_data + 2*cs[1] + valh);
                    hSign = -1; // Flip sign
                }

                // Read data from vertical neighbours (on same column)
                float vSign = 1;
                float xv = 0, yv = 0, zv = 0;
                if (r-1 >= 0)
                {
                    long valv = b*cs[0] + (r-1)*cs[2] + c*cs[3]; // Don't add stride along 3D dim
                    xv = *(cloud1_data + 0*cs[1] + valv);
                    yv = *(cloud1_data + 1*cs[1] + valv);
                    zv = *(cloud1_data + 2*cs[1] + valv);
                }
                if ((zv == 0) && (r+1 < nrows))// if r-1 is out of limits this will trigger
                {
                    long valv = b*cs[0] + (r+1)*cs[2] + c*cs[3]; // Don't add stride along 3D dim
                    xv = *(cloud1_data + 0*cs[1] + valv);
                    yv = *(cloud1_data + 1*cs[1] + valv);
                    zv = *(cloud1_data + 2*cs[1] + valv);
                    vSign = -1; // Flip sign
                }

                // Check max/min diff in depth. If it is crossing an edge then don't use that point
                if (fabs(zh - zc) > maxdepthdiff)
                {
                    xh = 0; yh = 0; zh = 0; // Set to zero
                }
                if (fabs(zv - zc) > maxdepthdiff)
                {
                    xv = 0; yv = 0; zv = 0; // Set to zero
                }

                // Compute the normal
                float nx, ny, nz;
                if (zh > 0 && zv > 0 && zc > 0)
                {
                    // Horizontal vector
                    float u1 = hSign * (xh - xc);
                    float u2 = hSign * (yh - yc);
                    float u3 = hSign * (zh - zc);

                    // Vertical vector
                    float v1 = vSign * (xv - xc);
                    float v2 = vSign * (yv - yc);
                    float v3 = vSign * (zv - zc);

                    // Compute cross product
                    nx = (u2*v3 - u3*v2);
                    ny = (u3*v1 - u1*v3);
                    nz = (u1*v2 - u2*v1);

                    // Normalize the vector to get unit direction
                    const float nMagSquared = nx*nx + ny*ny + nz*nz;
                    if (nMagSquared)
                    {
                        float s = 1.0/sqrtf(nMagSquared);
                        nx *= s; ny *= s; nz *= s; // Normalize to unit norm
                    }
                    else
                    {
                        nx = 0; ny = 0; nz = 0;
                    }

                }
                else
                {
                    nx = 0; ny = 0; nz = 0;
                }

                // Save normal
                *(normals1_data + 0*cs[1] + valc) = nx;
                *(normals1_data + 1*cs[1] + valc) = ny;
                *(normals1_data + 2*cs[1] + valc) = nz;

                /// === Compute transformed normal
                // Get delta transform for link of that point
                unsigned char l1 = *(label1_data + b*ls[0] + r*ls[2] + c*ls[3]);
                const float *T1  = deltaposes12_data + b*ps[0] + l1*ps[1]; // Get the 'l1'th transform

                // Transformed normal = R * orig normal (no translation)
                *(tnormals2_data + 0*cs[1] + valc) = T1[0] * nx + T1[1] * ny + T1[2]  * nz;
                *(tnormals2_data + 1*cs[1] + valc) = T1[4] * nx + T1[5] * ny + T1[6]  * nz;
                *(tnormals2_data + 2*cs[1] + valc) = T1[8] * nx + T1[9] * ny + T1[10] * nz;
            }
        }
    }

    /// ========= Free created memory
    THFloatTensor_free(cloud_1);
    THFloatTensor_free(cloud_2);
    THByteTensor_free(label_1);
    THFloatTensor_free(deltaposes_12);
    THFloatTensor_free(normals_1);
    THFloatTensor_free(tnormals_2);

    // Return
    return 1;
}
