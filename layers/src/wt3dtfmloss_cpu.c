#include <TH/TH.h>
#include <assert.h>
#include <stdbool.h>

// ===== FLOAT DATA

float Weighted3DTransformLoss_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetpoints,
            int sizeAverage)
{
    // Initialize vars
    long batchSize = points->size[0];
    //long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // New memory in case the inputs are not contiguous
    points = THFloatTensor_newContiguous(points);
    masks  = THFloatTensor_newContiguous(masks);
    tfms   = THFloatTensor_newContiguous(tfms);
    targetpoints = THFloatTensor_newContiguous(targetpoints);

    // Get data pointers
    float *points_data    = THFloatTensor_data(points);
    float *masks_data 	  = THFloatTensor_data(masks);
    float *tfms_data      = THFloatTensor_data(tfms);
    float *targetpts_data = THFloatTensor_data(targetpoints);

    // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

    // Iterate over all points
    double loss = 0.0; // Initialize loss
    long b,k,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get input point (p)
                long valp = b*ps[0] + r*ps[2] + c*ps[3]; // Don't add stride along 3D dim
                float x  = *(points_data + 0*ps[1] + valp);
                float y  = *(points_data + 1*ps[1] + valp);
                float z  = *(points_data + 2*ps[1] + valp);

                // Get target point (pt)
                float xt = *(targetpts_data + 0*ps[1] + valp);
                float yt = *(targetpts_data + 1*ps[1] + valp);
                float zt = *(targetpts_data + 2*ps[1] + valp);

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                for (k = 0; k < nSE3; k++)
                {
                    // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
                    float *T = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform
                    float xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
                    float yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
                    float zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

                    // Compute 3D squared-error between target pts & predicted points
                    float err = (xp-xt)*(xp-xt) + (yp-yt)*(yp-yt) + (zp-zt)*(zp-zt); //pow(xp - xt, 2) + pow(yp - yt, 2) + pow(zp - zt, 2);

                    // Weight the error by the mask weight
                    float w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th component of the error
                    loss += w_k * err;
                }
            }
        }
    }

    // Divide by number of points if asked for average
    loss *= 0.5; // Scale loss by 0.5
    if(sizeAverage)
    {
        long nElements = THFloatTensor_nElement(points);
        loss /= ((double)nElements);
    }

    // Free memory
    THFloatTensor_free(points);
    THFloatTensor_free(masks);
    THFloatTensor_free(tfms);
    THFloatTensor_free(targetpoints);

    return (float)loss;
}

void Weighted3DTransformLoss_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetpoints,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
            int sizeAverage)
{
    // Initialize vars
    long batchSize = points->size[0];
    //long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // Set gradients w.r.t pts & tfms to zero (as we add to these in a loop later)
    THFloatTensor_fill(gradPoints, 0);
    THFloatTensor_fill(gradTfms, 0);

    // New memory in case the inputs are not contiguous
    points = THFloatTensor_newContiguous(points);
    masks  = THFloatTensor_newContiguous(masks);
    tfms   = THFloatTensor_newContiguous(tfms);
    targetpoints = THFloatTensor_newContiguous(targetpoints);

    // Get data pointers
    float *points_data        = THFloatTensor_data(points);
    float *masks_data         = THFloatTensor_data(masks);
    float *tfms_data          = THFloatTensor_data(tfms);
    float *gradPoints_data    = THFloatTensor_data(gradPoints);
    float *gradMasks_data     = THFloatTensor_data(gradMasks);
    float *gradTfms_data      = THFloatTensor_data(gradTfms);
    float *targetpts_data     = THFloatTensor_data(targetpoints);

    // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

    // Iterate over all points
    long b,k,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get input point (p)
                long valp = b*ps[0] + r*ps[2] + c*ps[3]; // Don't add stride along 3D dim
                float x  = *(points_data + 0*ps[1] + valp);
                float y  = *(points_data + 1*ps[1] + valp);
                float z  = *(points_data + 2*ps[1] + valp);

                // Get target point (pt)
                float xt = *(targetpts_data + 0*ps[1] + valp);
                float yt = *(targetpts_data + 1*ps[1] + valp);
                float zt = *(targetpts_data + 2*ps[1] + valp);

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                float gx = 0, gy = 0, gz = 0; // Grads w.r.t input pts
                for (k = 0; k < nSE3; k++)
                {
                    // Get transform & wt
                    float *T  = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform
                    float w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th component of the error (scale by 0.5)

                    // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
                    float xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
                    float yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
                    float zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

                    // Compute difference between pred & target
                    float xd = (xp - xt);
                    float yd = (yp - yt);
                    float zd = (zp - zt);

                    // === Gradient w.r.t input points (p)
                    // (p = w_k * R^T * diff, summed across all the "k" transforms)
                    gx += w_k * (T[0] * xd + T[4] * yd + T[8]  * zd);
                    gy += w_k * (T[1] * xd + T[5] * yd + T[9]  * zd);
                    gz += w_k * (T[2] * xd + T[6] * yd + T[10] * zd);

                    // === Gradient w.r.t mask (w_k) = 0.5 * err
                    *(gradMasks_data + k*ms[1] + valm) = 0.5 * (pow(xd, 2) + pow(yd, 2) + pow(zd, 2));

                    // === Gradients w.r.t transforms (t_k)
                    float *gT = gradTfms_data + b*ts[0] + k*ts[1]; // Get the gradient of the 'k'th transform

                    // Grads w.r.t rotation parameters (sum across all pts)
                    gT[0]  += w_k * x * xd;
                    gT[1]  += w_k * y * xd;
                    gT[2]  += w_k * z * xd;
                    gT[4]  += w_k * x * yd;
                    gT[5]  += w_k * y * yd;
                    gT[6]  += w_k * z * yd;
                    gT[8]  += w_k * x * zd;
                    gT[9]  += w_k * y * zd;
                    gT[10] += w_k * z * zd;

                    // Grads w.r.t translation parameters (sum across all pts)
                    gT[3]  += w_k * xd;
                    gT[7]  += w_k * yd;
                    gT[11] += w_k * zd;
                }

                // Save gradients w.r.t points
                *(gradPoints_data + 0*ps[1] + valp) = gx;
                *(gradPoints_data + 1*ps[1] + valp) = gy;
                *(gradPoints_data + 2*ps[1] + valp) = gz;
            }
        }
    }

    // Average the gradients if sizeaverage is set
    if (sizeAverage)
    {
        // Compute scale factor
        long nElements = THFloatTensor_nElement(points);
        float norm = 1.0/((float)nElements);

        // Average gradients
        THFloatTensor_mul(gradPoints, gradPoints, norm);
        THFloatTensor_mul(gradMasks, gradMasks, norm);
        THFloatTensor_mul(gradTfms, gradTfms, norm);
    }

    // Free memory
    THFloatTensor_free(points);
    THFloatTensor_free(masks);
    THFloatTensor_free(tfms);
    THFloatTensor_free(targetpoints);
}

// ===== DOUBLE DATA

double Weighted3DTransformLoss_forward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetpoints,
            int sizeAverage)
{
    // Initialize vars
    long batchSize = points->size[0];
    //long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // New memory in case the inputs are not contiguous
    points = THDoubleTensor_newContiguous(points);
    masks  = THDoubleTensor_newContiguous(masks);
    tfms   = THDoubleTensor_newContiguous(tfms);
    targetpoints = THDoubleTensor_newContiguous(targetpoints);

    // Get data pointers
    double *points_data    = THDoubleTensor_data(points);
    double *masks_data 	  = THDoubleTensor_data(masks);
    double *tfms_data      = THDoubleTensor_data(tfms);
    double *targetpts_data = THDoubleTensor_data(targetpoints);

    // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

    // Iterate over all points
    double loss = 0.0; // Initialize loss
    long b,k,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get input point (p)
                long valp = b*ps[0] + r*ps[2] + c*ps[3]; // Don't add stride along 3D dim
                double x  = *(points_data + 0*ps[1] + valp);
                double y  = *(points_data + 1*ps[1] + valp);
                double z  = *(points_data + 2*ps[1] + valp);

                // Get target point (pt)
                double xt = *(targetpts_data + 0*ps[1] + valp);
                double yt = *(targetpts_data + 1*ps[1] + valp);
                double zt = *(targetpts_data + 2*ps[1] + valp);

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                for (k = 0; k < nSE3; k++)
                {
                    // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
                    double *T = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform
                    double xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
                    double yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
                    double zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

                    // Compute 3D squared-error between target pts & predicted points
                    double err = (xp-xt)*(xp-xt) + (yp-yt)*(yp-yt) + (zp-zt)*(zp-zt); //pow(xp - xt, 2) + pow(yp - yt, 2) + pow(zp - zt, 2);

                    // Weight the error by the mask weight
                    double w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th component of the error
                    loss += w_k * err;
                }
            }
        }
    }

    // Divide by number of points if asked for average
    loss *= 0.5; // Scale loss by 0.5
    if(sizeAverage)
    {
        long nElements = THDoubleTensor_nElement(points);
        loss /= ((double)nElements);
    }

    // Free memory
    THDoubleTensor_free(points);
    THDoubleTensor_free(masks);
    THDoubleTensor_free(tfms);
    THDoubleTensor_free(targetpoints);

    return loss;
}

void Weighted3DTransformLoss_backward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetpoints,
			THDoubleTensor *gradPoints,
			THDoubleTensor *gradMasks,
			THDoubleTensor *gradTfms,
            int sizeAverage)
{
    // Initialize vars
    long batchSize = points->size[0];
    //long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // Set gradients w.r.t pts & tfms to zero (as we add to these in a loop later)
    THDoubleTensor_fill(gradPoints, 0);
    THDoubleTensor_fill(gradTfms, 0);

    // New memory in case the inputs are not contiguous
    points = THDoubleTensor_newContiguous(points);
    masks  = THDoubleTensor_newContiguous(masks);
    tfms   = THDoubleTensor_newContiguous(tfms);
    targetpoints = THDoubleTensor_newContiguous(targetpoints);

    // Get data pointers
    double *points_data        = THDoubleTensor_data(points);
    double *masks_data         = THDoubleTensor_data(masks);
    double *tfms_data          = THDoubleTensor_data(tfms);
    double *gradPoints_data    = THDoubleTensor_data(gradPoints);
    double *gradMasks_data     = THDoubleTensor_data(gradMasks);
    double *gradTfms_data      = THDoubleTensor_data(gradTfms);
    double *targetpts_data     = THDoubleTensor_data(targetpoints);

    // Get strides
    long *ps = points->stride;
    long *ms = masks->stride;
    long *ts = tfms->stride;

    // Iterate over all points
    long b,k,r,c;
    for(b = 0; b < batchSize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get input point (p)
                long valp = b*ps[0] + r*ps[2] + c*ps[3]; // Don't add stride along 3D dim
                double x  = *(points_data + 0*ps[1] + valp);
                double y  = *(points_data + 1*ps[1] + valp);
                double z  = *(points_data + 2*ps[1] + valp);

                // Get target point (pt)
                double xt = *(targetpts_data + 0*ps[1] + valp);
                double yt = *(targetpts_data + 1*ps[1] + valp);
                double zt = *(targetpts_data + 2*ps[1] + valp);

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                double gx = 0, gy = 0, gz = 0; // Grads w.r.t input pts
                for (k = 0; k < nSE3; k++)
                {
                    // Get transform & wt
                    double *T  = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform
                    double w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th component of the error (scale by 0.5)

                    // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
                    double xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
                    double yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
                    double zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

                    // Compute difference between pred & target
                    double xd = (xp - xt);
                    double yd = (yp - yt);
                    double zd = (zp - zt);

                    // === Gradient w.r.t input points (p)
                    // (p = w_k * R^T * diff, summed across all the "k" transforms)
                    gx += w_k * (T[0] * xd + T[4] * yd + T[8]  * zd);
                    gy += w_k * (T[1] * xd + T[5] * yd + T[9]  * zd);
                    gz += w_k * (T[2] * xd + T[6] * yd + T[10] * zd);

                    // === Gradient w.r.t mask (w_k) = 0.5 * err
                    *(gradMasks_data + k*ms[1] + valm) = 0.5 * (pow(xd, 2) + pow(yd, 2) + pow(zd, 2));

                    // === Gradients w.r.t transforms (t_k)
                    double *gT = gradTfms_data + b*ts[0] + k*ts[1]; // Get the gradient of the 'k'th transform

                    // Grads w.r.t rotation parameters (sum across all pts)
                    gT[0]  += w_k * x * xd;
                    gT[1]  += w_k * y * xd;
                    gT[2]  += w_k * z * xd;
                    gT[4]  += w_k * x * yd;
                    gT[5]  += w_k * y * yd;
                    gT[6]  += w_k * z * yd;
                    gT[8]  += w_k * x * zd;
                    gT[9]  += w_k * y * zd;
                    gT[10] += w_k * z * zd;

                    // Grads w.r.t translation parameters (sum across all pts)
                    gT[3]  += w_k * xd;
                    gT[7]  += w_k * yd;
                    gT[11] += w_k * zd;
                }

                // Save gradients w.r.t points
                *(gradPoints_data + 0*ps[1] + valp) = gx;
                *(gradPoints_data + 1*ps[1] + valp) = gy;
                *(gradPoints_data + 2*ps[1] + valp) = gz;
            }
        }
    }

    // Average the gradients if sizeaverage is set
    if (sizeAverage)
    {
        // Compute scale factor
        long nElements = THDoubleTensor_nElement(points);
        double norm = 1.0/((float)nElements);

        // Average gradients
        THDoubleTensor_mul(gradPoints, gradPoints, norm);
        THDoubleTensor_mul(gradMasks, gradMasks, norm);
        THDoubleTensor_mul(gradTfms, gradTfms, norm);
    }

    // Free memory
    THDoubleTensor_free(points);
    THDoubleTensor_free(masks);
    THDoubleTensor_free(tfms);
    THDoubleTensor_free(targetpoints);
}