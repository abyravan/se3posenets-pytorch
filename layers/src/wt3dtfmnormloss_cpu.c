#include <TH/TH.h>
#include <assert.h>
#include <stdbool.h>

// ===== FLOAT DATA

float Weighted3DTransformNormLoss_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetflows,
            THFloatTensor *numpts,
			float normWt,
			int normPerPt)
{
    // Initialize vars
    long batchSize = points->size[0];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // New memory in case the inputs are not contiguous
    points = THFloatTensor_newContiguous(points);
    masks  = THFloatTensor_newContiguous(masks);
    tfms   = THFloatTensor_newContiguous(tfms);
    targetflows = THFloatTensor_newContiguous(targetflows);
    numpts = THFloatTensor_newContiguous(numpts);

    // Get data pointers
    float *points_data      = THFloatTensor_data(points);
    float *masks_data 	    = THFloatTensor_data(masks);
    float *tfms_data        = THFloatTensor_data(tfms);
    float *targetflows_data = THFloatTensor_data(targetflows);
    float *numpts_data      = THFloatTensor_data(numpts);

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

                // Get target flow (ft)
                float fxt = *(targetflows_data + 0*ps[1] + valp);
                float fyt = *(targetflows_data + 1*ps[1] + valp);
                float fzt = *(targetflows_data + 2*ps[1] + valp);

                // Get normalizing constant (sigma), clamped to a min of 2e-3
                float sx = 0, sy = 0, sz = 0, s = 0;
                if (normPerPt)
                {
                    s = fmaxf(normWt * pow(fxt*fxt + fyt*fyt + fzt*fzt, 0.5), 2e-3); // Scale by length of flow vector
                }
                else
                {
                    // Independent per dimension of the flow vector
                    sx = fmaxf(normWt * fabsf(fxt), 2e-3);
                    sy = fmaxf(normWt * fabsf(fyt), 2e-3);
                    sz = fmaxf(normWt * fabsf(fzt), 2e-3);
                }

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                for (k = 0; k < nSE3; k++)
                {
                    // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
                    float *T = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform
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
                    float w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th component of the error
                    loss += w_k * err / numpts_data[b];
                }
            }
        }
    }

    // Divide by number of points if asked for average
    loss /= (2.0 * ((double) batchSize));

    // Free memory
    THFloatTensor_free(points);
    THFloatTensor_free(masks);
    THFloatTensor_free(tfms);
    THFloatTensor_free(targetflows);

    return (float)loss;
}

void Weighted3DTransformNormLoss_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetflows,
            THFloatTensor *numpts,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
            THFloatTensor *gradOutput,
            float normWt,
            int normPerPt)
{
    // Initialize vars
    long batchSize = points->size[0];
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
    targetflows = THFloatTensor_newContiguous(targetflows);
    numpts = THFloatTensor_newContiguous(numpts);

    // Get data pointers
    float *points_data        = THFloatTensor_data(points);
    float *masks_data         = THFloatTensor_data(masks);
    float *tfms_data          = THFloatTensor_data(tfms);
    float *gradPoints_data    = THFloatTensor_data(gradPoints);
    float *gradMasks_data     = THFloatTensor_data(gradMasks);
    float *gradTfms_data      = THFloatTensor_data(gradTfms);
    float *targetflows_data   = THFloatTensor_data(targetflows);
    float *gradOutput_data    = THFloatTensor_data(gradOutput);
    float *numpts_data        = THFloatTensor_data(numpts);

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

                // Get target flow (fp)
                float fxt = *(targetflows_data + 0*ps[1] + valp);
                float fyt = *(targetflows_data + 1*ps[1] + valp);
                float fzt = *(targetflows_data + 2*ps[1] + valp);

                // Get normalizing constant (sigma), clamped to a min of 2e-3
                float sx = 0, sy = 0, sz = 0, s = 0;
                if (normPerPt)
                {
                    s = fmaxf(normWt * pow(fxt*fxt + fyt*fyt + fzt*fzt, 0.5), 2e-3); // Scale by length of flow vector
                }
                else
                {
                    // Independent per dimension of the flow vector
                    sx = fmaxf(normWt * fabsf(fxt), 2e-3);
                    sy = fmaxf(normWt * fabsf(fyt), 2e-3);
                    sz = fmaxf(normWt * fabsf(fzt), 2e-3);
                }

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
                    *(gradMasks_data + k*ms[1] + valm) = 0.5*err / numpts_data[b];

                    // == Scale error terms by sigma (from here on we only use the scaled terms)
                    ex /= normPerPt ? s : sx;
                    ey /= normPerPt ? s : sy;
                    ez /= normPerPt ? s : sz;

                    // === Gradient w.r.t input points (p)
                    // (p = w_k * (R^T - I) * diff/sigma, summed across all the "k" transforms)
                    gx += w_k * ((T[0]-1.0) * ex + T[4]       * ey + T[8]        * ez);
                    gy += w_k * (T[1]       * ex + (T[5]-1.0) * ey + T[9]        * ez);
                    gz += w_k * (T[2]       * ex + T[6]       * ey + (T[10]-1.0) * ez);

                    // === Gradients w.r.t transforms (t_k)
                    float *gT = gradTfms_data + b*ts[0] + k*ts[1]; // Get the gradient of the 'k'th transform

                    // Scale by numpts
                    ex /= numpts_data[b];
                    ey /= numpts_data[b];
                    ez /= numpts_data[b];

                    // Grads w.r.t rotation parameters (sum across all pts)
                    gT[0]  += w_k * x * ex;
                    gT[1]  += w_k * y * ex;
                    gT[2]  += w_k * z * ex;
                    gT[4]  += w_k * x * ey;
                    gT[5]  += w_k * y * ey;
                    gT[6]  += w_k * z * ey;
                    gT[8]  += w_k * x * ez;
                    gT[9]  += w_k * y * ez;
                    gT[10] += w_k * z * ez;

                    // Grads w.r.t translation parameters (sum across all pts)
                    gT[3]  += w_k * ex;
                    gT[7]  += w_k * ey;
                    gT[11] += w_k * ez;
                }

                // Save gradients w.r.t points
                *(gradPoints_data + 0*ps[1] + valp) = gx / numpts_data[b];
                *(gradPoints_data + 1*ps[1] + valp) = gy / numpts_data[b];
                *(gradPoints_data + 2*ps[1] + valp) = gz / numpts_data[b];
            }
        }
    }

    // Scale by grad output & average gradients
    float norm = gradOutput_data[0] / (float) batchSize;
    THFloatTensor_mul(gradPoints, gradPoints, norm);
    THFloatTensor_mul(gradMasks, gradMasks, norm);
    THFloatTensor_mul(gradTfms, gradTfms, norm);

    // Free memory
    THFloatTensor_free(points);
    THFloatTensor_free(masks);
    THFloatTensor_free(tfms);
    THFloatTensor_free(targetflows);
}

// ===== DOUBLE DATA

double Weighted3DTransformNormLoss_forward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetflows,
            THDoubleTensor *numpts,
			float normWt,
			int normPerPt)
{
    // Initialize vars
    long batchSize = points->size[0];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // New memory in case the inputs are not contiguous
    points = THDoubleTensor_newContiguous(points);
    masks  = THDoubleTensor_newContiguous(masks);
    tfms   = THDoubleTensor_newContiguous(tfms);
    targetflows = THDoubleTensor_newContiguous(targetflows);
    numpts = THDoubleTensor_newContiguous(numpts);

    // Get data pointers
    double *points_data      = THDoubleTensor_data(points);
    double *masks_data 	     = THDoubleTensor_data(masks);
    double *tfms_data        = THDoubleTensor_data(tfms);
    double *targetflows_data = THDoubleTensor_data(targetflows);
    double *numpts_data      = THDoubleTensor_data(numpts);

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

                // Get target flow (ft)
                double fxt = *(targetflows_data + 0*ps[1] + valp);
                double fyt = *(targetflows_data + 1*ps[1] + valp);
                double fzt = *(targetflows_data + 2*ps[1] + valp);

                // Get normalizing constant (sigma), clamped to a min of 2e-3
                double sx = 0, sy = 0, sz = 0, s = 0;
                if (normPerPt)
                {
                    s = fmax(normWt * pow(fxt*fxt + fyt*fyt + fzt*fzt, 0.5), 2e-3); // Scale by length of flow vector
                }
                else
                {
                    // Independent per dimension of the flow vector
                    sx = fmax(normWt * fabs(fxt), 2e-3);
                    sy = fmax(normWt * fabs(fyt), 2e-3);
                    sz = fmax(normWt * fabs(fzt), 2e-3);
                }

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                for (k = 0; k < nSE3; k++)
                {
                    // Compute transformed 3D point: p' = (R_k*p + t_k) (for X,Y,Z coordinates)
                    double *T = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform
                    double xp = (T[0] * x + T[1] * y + T[2]  * z + T[3]);  // (R_k * p_x + t_k)
                    double yp = (T[4] * x + T[5] * y + T[6]  * z + T[7]);  // (R_k * p_y + t_k)
                    double zp = (T[8] * x + T[9] * y + T[10] * z + T[11]); // (R_k * p_z + t_k)

                    // Compute flow error (predicted - target flow)
                    double ex = (xp - x) - fxt;
                    double ey = (yp - y) - fyt;
                    double ez = (zp - z) - fzt;

                    // Compute normalized error (different scalar per dimension)
                    double err;
                    if (normPerPt)
                        err = (ex*ex + ey*ey + ez*ez) / s;
                    else
                        err = (ex*ex)/sx + (ey*ey)/sy + (ez*ez)/sz; // different scale per dimension

                    // Weight the error by the mask weight
                    double w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th component of the error
                    loss += w_k * err / numpts_data[b];
                }
            }
        }
    }

    // Divide by number of points if asked for average
    loss /= (2.0 * ((double) batchSize));

    // Free memory
    THDoubleTensor_free(points);
    THDoubleTensor_free(masks);
    THDoubleTensor_free(tfms);
    THDoubleTensor_free(targetflows);
    THDoubleTensor_free(numpts);

    return loss;
}

void Weighted3DTransformNormLoss_backward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetflows,
            THDoubleTensor *numpts,
			THDoubleTensor *gradPoints,
			THDoubleTensor *gradMasks,
			THDoubleTensor *gradTfms,
            THDoubleTensor *gradOutput,
            float normWt,
            int normPerPt)
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
    targetflows = THDoubleTensor_newContiguous(targetflows);
    numpts = THDoubleTensor_newContiguous(numpts);

    // Get data pointers
    double *points_data        = THDoubleTensor_data(points);
    double *masks_data         = THDoubleTensor_data(masks);
    double *tfms_data          = THDoubleTensor_data(tfms);
    double *gradPoints_data    = THDoubleTensor_data(gradPoints);
    double *gradMasks_data     = THDoubleTensor_data(gradMasks);
    double *gradTfms_data      = THDoubleTensor_data(gradTfms);
    double *targetflows_data   = THDoubleTensor_data(targetflows);
    double *gradOutput_data    = THDoubleTensor_data(gradOutput);
    double *numpts_data        = THDoubleTensor_data(numpts);

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

                // Get target flow (fp)
                double fxt = *(targetflows_data + 0*ps[1] + valp);
                double fyt = *(targetflows_data + 1*ps[1] + valp);
                double fzt = *(targetflows_data + 2*ps[1] + valp);

                // Get normalizing constant (sigma), clamped to a min of 2e-3
                double sx = 0, sy = 0, sz = 0, s = 0;
                if (normPerPt)
                {
                    s = fmax(normWt * pow(fxt*fxt + fyt*fyt + fzt*fzt, 0.5), 2e-3); // Scale by length of flow vector
                }
                else
                {
                    // Independent per dimension of the flow vector
                    sx = fmax(normWt * fabs(fxt), 2e-3);
                    sy = fmax(normWt * fabs(fyt), 2e-3);
                    sz = fmax(normWt * fabs(fzt), 2e-3);
                }

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

                    // Compute flow error (predicted - target flow)
                    double ex = (xp - x) - fxt;
                    double ey = (yp - y) - fyt;
                    double ez = (zp - z) - fzt;

                    // === Gradient w.r.t mask (w_k) = 0.5 * err
                    double err;
                    if (normPerPt)
                        err = (ex*ex + ey*ey + ez*ez) / s;
                    else
                        err = (ex*ex)/sx + (ey*ey)/sy + (ez*ez)/sz; // different scale per dimension
                    *(gradMasks_data + k*ms[1] + valm) = 0.5*err / numpts_data[b];

                    // == Scale error terms by sigma (from here on we only use the scaled terms)
                    ex /= normPerPt ? s : sx;
                    ey /= normPerPt ? s : sy;
                    ez /= normPerPt ? s : sz;

                    // === Gradient w.r.t input points (p)
                    // (p = w_k * (R^T - I) * diff/sigma, summed across all the "k" transforms)
                    gx += w_k * ((T[0]-1.0) * ex + T[4]       * ey + T[8]        * ez);
                    gy += w_k * (T[1]       * ex + (T[5]-1.0) * ey + T[9]        * ez);
                    gz += w_k * (T[2]       * ex + T[6]       * ey + (T[10]-1.0) * ez);

                    // === Gradients w.r.t transforms (t_k)
                    double *gT = gradTfms_data + b*ts[0] + k*ts[1]; // Get the gradient of the 'k'th transform

                    // Scale by numpts
                    ex /= numpts_data[b];
                    ey /= numpts_data[b];
                    ez /= numpts_data[b];

                    // Grads w.r.t rotation parameters (sum across all pts)
                    gT[0]  += w_k * x * ex;
                    gT[1]  += w_k * y * ex;
                    gT[2]  += w_k * z * ex;
                    gT[4]  += w_k * x * ey;
                    gT[5]  += w_k * y * ey;
                    gT[6]  += w_k * z * ey;
                    gT[8]  += w_k * x * ez;
                    gT[9]  += w_k * y * ez;
                    gT[10] += w_k * z * ez;

                    // Grads w.r.t translation parameters (sum across all pts)
                    gT[3]  += w_k * ex;
                    gT[7]  += w_k * ey;
                    gT[11] += w_k * ez;
                }

                // Save gradients w.r.t points
                *(gradPoints_data + 0*ps[1] + valp) = gx / numpts_data[b];
                *(gradPoints_data + 1*ps[1] + valp) = gy / numpts_data[b];
                *(gradPoints_data + 2*ps[1] + valp) = gz / numpts_data[b];
            }
        }
    }

    // Scale by grad output & average gradients
    double norm = gradOutput_data[0] / (double) batchSize;
    THDoubleTensor_mul(gradPoints, gradPoints, norm);
    THDoubleTensor_mul(gradMasks, gradMasks, norm);
    THDoubleTensor_mul(gradTfms, gradTfms, norm);

    // Free memory
    THDoubleTensor_free(points);
    THDoubleTensor_free(masks);
    THDoubleTensor_free(tfms);
    THDoubleTensor_free(targetflows);
    THDoubleTensor_free(numpts);
}