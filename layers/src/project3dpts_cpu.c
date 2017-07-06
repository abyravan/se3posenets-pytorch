#include <TH/TH.h>
#include <assert.h>
#include <stdbool.h>

// ===== FLOAT DATA

bool perspectiveProjectionToPixel_float(const float x,  const float y,  const float z,
                                        const float fx, const float fy,
                                        const float cx, const float cy,
                                        float *xpix, float *ypix)
{
    // Do a perspective transform, scale by focal length & add principal point
    // TODO: Check the +1 here
    if (z > 0)
    {
        (*xpix) = ((x/z) * fx) + cx;// + 1; // Points go from [0, row-1] & [0, col-1] in original data
        (*ypix) = ((y/z) * fy) + cy;// + 1;
        return true;
    }
    return false;
}

int Project3DPointsToSubPixelDepth_forward_float(
                                                 THFloatTensor *input,
                                                 THFloatTensor *indexMap,
                                                 THFloatTensor *output,
                                                 float fy, float fx,
                                                 float cy, float cx)
{
    // Initialize vars
    long batchSize      = input->size[0];
    long nrows          = input->size[2];
    long ncols          = input->size[3];

    // Fill with defaults
    THFloatTensor_fill(output, 0);
    THFloatTensor_fill(indexMap, -1); // -1 means invalid

    // Get data pointers
    float *input_data 		= THFloatTensor_data(input);
    float *output_data 		= THFloatTensor_data(output);
    float *indexMap_data 	= THFloatTensor_data(indexMap);

    // Get strides
    long *is    = input->stride;
    long *iMs   = indexMap->stride;
    long *os    = output->stride;

    // Iterate over all points
    long b,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get input point
                long vali = b*is[0] + r*is[2] + c*is[3]; // Don't add stride along 3D dim
                float x = *(input_data + 0*is[1] + vali);
                float y = *(input_data + 1*is[1] + vali);
                float z = *(input_data + 2*is[1] + vali);

                // Compute pixel coordinates based on focal length and COP
                float xpix, ypix;
                bool success = perspectiveProjectionToPixel_float(x,y,z,fx,fy,cx,cy,&xpix,&ypix);
                if(!success) continue; // No valid projection : Z <= 0

                // Check projection success / Check limits / Do the depth test
                float xpixr = round(xpix); // Rounded off pixel col
                float ypixr = round(ypix); // Rounded off pixel row
                if (xpixr >= 0 && xpixr < ncols && ypixr >= 0 && ypixr < nrows)
                {
                    // Do depth test:
                    //   If z >= z at pixel, discard this point
                    //   Else z at pixel = z
                    long valo = b*os[0] + ypixr*os[2] + xpixr*os[3]; // y = row, x = col
                    float zo   = *(output_data + 2*os[1] + valo);
                    if ((zo == 0) || (z < zo))
                    {
                        // (XO,YO,ZO) = (xpix, ypix, z) ==> Save sub-pixel x,y and depth
                        *(output_data + 0*os[1] + valo) = xpix;
                        *(output_data + 1*os[1] + valo) = ypix;
                        *(output_data + 2*os[1] + valo) = z;

                        // Update the index map
                        long valim = b*iMs[0] + ypixr*iMs[2] + xpixr*iMs[3]; // y = row, x = col
                        *(indexMap_data + valim) = vali; // ID of input point for that pixel
                    }
                }
            }
        }
    }

    // Return value
    return 1;
}

int Project3DPointsToSubPixelDepth_backward_float(
                                                  THFloatTensor *input,
                                                  THFloatTensor *indexMap,
                                                  THFloatTensor *gradInput,
                                                  THFloatTensor *gradOutput,
                                                  float fy, float fx,
                                                  float cy, float cx)
{
    // Initialize vars
    long batchSize      = input->size[0];
    long nrows          = input->size[2];
    long ncols          = input->size[3];

    // Fill with defaults
    THFloatTensor_fill(gradInput, 0);

    // Get data pointers
    float *input_data 		= THFloatTensor_data(input);
    float *gradOutput_data 	= THFloatTensor_data(gradOutput);
    float *indexMap_data 	= THFloatTensor_data(indexMap);
    float *gradInput_data 	= THFloatTensor_data(gradInput);

    // Get strides
    long *is    = input->stride;
    long *gos   = gradOutput->stride;
    long *iMs   = indexMap->stride;

    // Iterate over all output pixels
    long b,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get the index map value (for that output pixel)
                long valim = b*iMs[0] + r*iMs[2] + c*iMs[3]; // y = row, x = col
                long vali  = (long)(*(indexMap_data + valim));
                if (vali == -1) continue; // In case this point has no corresponding output index, continue

                // Get input point (from set of all input points)
                float x = *(input_data + 0*is[1] + vali);
                float y = *(input_data + 1*is[1] + vali);
                float z = *(input_data + 2*is[1] + vali);

                // Get gradOutput value (for that output pixel)
                long valgo = b*gos[0] + r*gos[2] + c*gos[3]; // y = row, x = col
                float gx = *(gradOutput_data + 0*is[1] + valgo);
                float gy = *(gradOutput_data + 1*is[1] + valgo);
                float gz = *(gradOutput_data + 2*is[1] + valgo);

                // Gradient w.r.t x = (fx/z) * gx
                // Gradient w.r.t y = (fy/z) * gy
                // Gradient w.r.t z = (-x/z^2) * fx * gx + (-y/z^2) * fy * gy + gz
                *(gradInput_data + 0*is[1] + vali) = (fx/z) * gx;
                *(gradInput_data + 1*is[1] + vali) = (fy/z) * gy;
                *(gradInput_data + 2*is[1] + vali) = ((-x/pow(z,2)) * fx * gx) + ((-y/pow(z,2)) * fy * gy) + gz;
            }
        }
    }

    return 1;
}

// ===== DOUBLE DATA

bool perspectiveProjectionToPixel_double(const double x,  const double y,  const double z,
                                         const double fx, const double fy,
                                         const double cx, const double cy,
                                         double *xpix, double *ypix)
{
    // Do a perspective transform, scale by focal length & add principal point
    // TODO: Check the +1 here
    if (z > 0)
    {
        (*xpix) = ((x/z) * fx) + cx;// + 1; // Points go from [0, row-1] & [0, col-1] in original data
        (*ypix) = ((y/z) * fy) + cy;// + 1;
        return true;
    }
    return false;
}

int Project3DPointsToSubPixelDepth_forward_double(
                                                  THDoubleTensor *input,
                                                  THDoubleTensor *indexMap,
                                                  THDoubleTensor *output,
                                                  double fy, double fx,
                                                  double cy, double cx)
{
    // Initialize vars
    long batchSize      = input->size[0];
    long nrows          = input->size[2];
    long ncols          = input->size[3];

    // Fill with defaults
    THDoubleTensor_fill(output, 0);
    THDoubleTensor_fill(indexMap, -1); // -1 means invalid

    // Get data pointers
    double *input_data 		= THDoubleTensor_data(input);
    double *output_data 	= THDoubleTensor_data(output);
    double *indexMap_data 	= THDoubleTensor_data(indexMap);

    // Get strides
    long *is    = input->stride;
    long *iMs   = indexMap->stride;
    long *os    = output->stride;

    // Iterate over all points
    long b,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get input point
                long vali = b*is[0] + r*is[2] + c*is[3]; // Don't add stride along 3D dim
                double x = *(input_data + 0*is[1] + vali);
                double y = *(input_data + 1*is[1] + vali);
                double z = *(input_data + 2*is[1] + vali);

                // Compute pixel coordinates based on focal length and COP
                double xpix, ypix;
                bool success = perspectiveProjectionToPixel_double(x,y,z,fx,fy,cx,cy,&xpix,&ypix);
                if(!success) continue; // No valid projection : Z <= 0

                // Check projection success / Check limits / Do the depth test
                double xpixr = round(xpix); // Rounded off pixel col
                double ypixr = round(ypix); // Rounded off pixel row
                if (xpixr >= 0 && xpixr < ncols && ypixr >= 0 && ypixr < nrows)
                {
                    // Do depth test:
                    //   If z >= z at pixel, discard this point
                    //   Else z at pixel = z
                    long valo = b*os[0] + ypixr*os[2] + xpixr*os[3]; // y = row, x = col
                    double zo   = *(output_data + 2*os[1] + valo);
                    if ((zo == 0) || (z < zo))
                    {
                        // (XO,YO,ZO) = (xpix, ypix, z) ==> Save sub-pixel x,y and depth
                        *(output_data + 0*os[1] + valo) = xpix;
                        *(output_data + 1*os[1] + valo) = ypix;
                        *(output_data + 2*os[1] + valo) = z;

                        // Update the index map
                        long valim = b*iMs[0] + ypixr*iMs[2] + xpixr*iMs[3]; // y = row, x = col
                        *(indexMap_data + valim) = vali; // ID of input point for that pixel
                    }
                }
            }
        }
    }

    // Return value
    return 1;
}

int Project3DPointsToSubPixelDepth_backward_double(
                                                   THDoubleTensor *input,
                                                   THDoubleTensor *indexMap,
                                                   THDoubleTensor *gradInput,
                                                   THDoubleTensor *gradOutput,
                                                   double fy, double fx,
                                                   double cy, double cx)
{
    // Initialize vars
    long batchSize      = input->size[0];
    long nrows          = input->size[2];
    long ncols          = input->size[3];

    // Fill with defaults
    THDoubleTensor_fill(gradInput, 0);

    // Get data pointers
    double *input_data 		= THDoubleTensor_data(input);
    double *gradOutput_data = THDoubleTensor_data(gradOutput);
    double *indexMap_data 	= THDoubleTensor_data(indexMap);
    double *gradInput_data 	= THDoubleTensor_data(gradInput);

    // Get strides
    long *is    = input->stride;
    long *gos   = gradOutput->stride;
    long *iMs   = indexMap->stride;

    // Iterate over all output pixels
    long b,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get the index map value (for that output pixel)
                long valim = b*iMs[0] + r*iMs[2] + c*iMs[3]; // y = row, x = col
                long vali  = (long)(*(indexMap_data + valim));
                if (vali == -1) continue; // In case this point has no corresponding output index, continue

                // Get input point (from set of all input points)
                double x = *(input_data + 0*is[1] + vali);
                double y = *(input_data + 1*is[1] + vali);
                double z = *(input_data + 2*is[1] + vali);

                // Get gradOutput value (for that output pixel)
                long valgo = b*gos[0] + r*gos[2] + c*gos[3]; // y = row, x = col
                double gx = *(gradOutput_data + 0*is[1] + valgo);
                double gy = *(gradOutput_data + 1*is[1] + valgo);
                double gz = *(gradOutput_data + 2*is[1] + valgo);

                // Gradient w.r.t x = (fx/z) * gx
                // Gradient w.r.t y = (fy/z) * gy
                // Gradient w.r.t z = (-x/z^2) * fx * gx + (-y/z^2) * fy * gy + gz
                *(gradInput_data + 0*is[1] + vali) = (fx/z) * gx;
                *(gradInput_data + 1*is[1] + vali) = (fy/z) * gy;
                *(gradInput_data + 2*is[1] + vali) = ((-x/pow(z,2)) * fx * gx) + ((-y/pow(z,2)) * fy * gy) + gz;
            }
        }
    }

    return 1;
}