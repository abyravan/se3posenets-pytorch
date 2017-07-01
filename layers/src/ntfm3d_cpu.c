#include <TH/TH.h>
#include <assert.h>

int NTfm3D_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *tfmpoints)
{
    // Initialize vars
    long batchSize = points->size[0];
    long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];
	 assert(ndim == 3);

    // Resize output and set defaults
    THFloatTensor_resizeAs(tfmpoints, points);

    // Get data pointers
    float *points_data 	  = THFloatTensor_data(points);
    float *masks_data 	  = THFloatTensor_data(masks);
    float *tfms_data      = THFloatTensor_data(tfms);
    float *tfmpoints_data = THFloatTensor_data(tfmpoints);

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
                float x = *(points_data + 0*ps[1] + valp);
                float y = *(points_data + 1*ps[1] + valp);
                float z = *(points_data + 2*ps[1] + valp);

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                float xt = 0, yt = 0, zt = 0;
                for (k = 0; k < nSE3; k++)
                {
                    // Get transform & wt
                    float w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th transform "
                    float *T  = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform

                    // Add w_k * (R_k*p + t_k) (for X,Y,Z coordinates)
                    xt += w_k * (T[0] * x + T[1] * y + T[2]  * z + T[3]); // w_k * (R_k * p_x + t_k)
                    yt += w_k * (T[4] * x + T[5] * y + T[6]  * z + T[7]); // w_k * (R_k * p_y + t_k)
                    zt += w_k * (T[8] * x + T[9] * y + T[10] * z + T[11]); // w_k * (R_k * p_z + t_k)
                }

                // Copy to output
                *(tfmpoints_data + 0*ps[1] + valp) = xt;
                *(tfmpoints_data + 1*ps[1] + valp) = yt;
                *(tfmpoints_data + 2*ps[1] + valp) = zt;
            }
        }
    }

	 return 1;
}

int NTfm3D_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *tfmpoints,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
			THFloatTensor *gradTfmpoints)
{
    // Initialize vars
    long batchSize = points->size[0];
    long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];
	 assert(ndim == 3);

    // Set gradients w.r.t pts & tfms to zero (as we add to these in a loop later)
    THFloatTensor_fill(gradPoints, 0);
    THFloatTensor_fill(gradTfms, 0);

    // Get data pointers
    float *points_data        = THFloatTensor_data(points);
    float *masks_data         = THFloatTensor_data(masks);
    float *tfms_data          = THFloatTensor_data(tfms);
    float *gradPoints_data 	= THFloatTensor_data(gradPoints);
    float *gradMasks_data 	   = THFloatTensor_data(gradMasks);
    float *gradTfms_data      = THFloatTensor_data(gradTfms);
    float *gradTfmpoints_data = THFloatTensor_data(gradTfmpoints);
    //float *tfmpoints_data     = THFloatTensor_data(tfmpoints);

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
                float x = *(points_data + 0*ps[1] + valp);
                float y = *(points_data + 1*ps[1] + valp);
                float z = *(points_data + 2*ps[1] + valp);

                // Get gradient w.r.t output point (gpt)
                float gxt = *(gradTfmpoints_data + 0*ps[1] + valp);
                float gyt = *(gradTfmpoints_data + 1*ps[1] + valp);
                float gzt = *(gradTfmpoints_data + 2*ps[1] + valp);

                // Gradients w.r.t pts, masks & tfms
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                float gx = 0, gy = 0, gz = 0; // Grads w.r.t input pts
                for (k = 0; k < nSE3; k++)
                {
                    // Get transform & wt
                    float w_k = *(masks_data + k*ms[1] + valm);   // Get the weight for the 'k'th transform "
                    float *T  = tfms_data + b*ts[0] + k*ts[1];     // Get the 'k'th transform

                    // === Gradient w.r.t input point (p = R^T * gpt, summed across all the "k" transforms)
                    gx += w_k * (T[0] * gxt + T[4] * gyt + T[8]  * gzt);
                    gy += w_k * (T[1] * gxt + T[5] * gyt + T[9]  * gzt);
                    gz += w_k * (T[2] * gxt + T[6] * gyt + T[10] * gzt);

                    // === Gradient w.r.t mask (w_k) = (R_k^T * p + t_k) * gpt
                    *(gradMasks_data + k*ms[1] + valm) = gxt * (T[0] * x + T[1] * y + T[2]  * z + T[3]) +
                                                         gyt * (T[4] * x + T[5] * y + T[6]  * z + T[7]) +
                                                         gzt * (T[8] * x + T[9] * y + T[10] * z + T[11]);

                    // === Gradients w.r.t transforms (t_k)
                    float *gT = gradTfms_data + b*ts[0] + k*ts[1]; // Get the gradient of the 'k'th transform

                    // Grads w.r.t rotation parameters (sum across all pts)
                    gT[0]  += w_k * x * gxt;
                    gT[1]  += w_k * y * gxt;
                    gT[2]  += w_k * z * gxt;
                    gT[4]  += w_k * x * gyt;
                    gT[5]  += w_k * y * gyt;
                    gT[6]  += w_k * z * gyt;
                    gT[8]  += w_k * x * gzt;
                    gT[9]  += w_k * y * gzt;
                    gT[10] += w_k * z * gzt;

                    // Grads w.r.t translation parameters (sum across all pts)
                    gT[3]  += w_k * gxt;
                    gT[7]  += w_k * gyt;
                    gT[11] += w_k * gzt;
                }

                // Gradients w.r.t pts (copy after sum across tfms)
                *(gradPoints_data + 0*ps[1] + valp) = gx;
                *(gradPoints_data + 1*ps[1] + valp) = gy;
                *(gradPoints_data + 2*ps[1] + valp) = gz;
            }
        }
    }

	 return 1;
}
