#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/NTfm3D_cpu.c"
#else

#include <stdbool.h>

void THNN_(NTfm3D_updateOutput)(
			THNNState *state,
			THTensor *points,
			THTensor *masks,
			THTensor *tfms,
			THTensor *tfmpoints)
{
    // Initialize vars
    long batchSize = points->size[0];
    long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // Resize output and set defaults
    THTensor_(resizeAs)(tfmpoints, points);

    // Get data pointers
    real *points_data 	 = THTensor_(data)(points);
    real *masks_data 	 = THTensor_(data)(masks);
    real *tfms_data      = THTensor_(data)(tfms);
    real *tfmpoints_data = THTensor_(data)(tfmpoints);

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
                real x = *(points_data + 0*ps[1] + valp);
                real y = *(points_data + 1*ps[1] + valp);
                real z = *(points_data + 2*ps[1] + valp);

                // Compute sum_k w_k * (R_k*p + t_k) across the different SE3s
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                real xt = 0, yt = 0, zt = 0;
                for (k = 0; k < nSE3; k++)
                {
                    // Get transform & wt
                    real w_k = *(masks_data + k*ms[1] + valm); // Get the weight for the 'k'th transform "
                    real *T = tfms_data + b*ts[0] + k*ts[1];   // Get the 'k'th transform

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
}

void THNN_(NTfm3D_updateGradInput)(
			THNNState *state,
			THTensor *points,
			THTensor *masks,
			THTensor *tfms,
			THTensor *tfmpoints,
			THTensor *gradPoints,
			THTensor *gradMasks,
			THTensor *gradTfms,
			THTensor *gradTfmpoints)
{
    // Initialize vars
    long batchSize = points->size[0];
    long ndim      = points->size[1];
    long nrows     = points->size[2];
    long ncols     = points->size[3];
    long nSE3      = masks->size[1];

    // Set gradients w.r.t pts & tfms to zero (as we add to these in a loop later)
    THTensor_(fill)(gradPoints, 0);
    THTensor_(fill)(gradTfms, 0);

    // Get data pointers
    real *points_data        = THTensor_(data)(points);
    real *masks_data         = THTensor_(data)(masks);
    real *tfms_data          = THTensor_(data)(tfms);
    real *gradPoints_data 	 = THTensor_(data)(gradPoints);
    real *gradMasks_data 	 = THTensor_(data)(gradMasks);
    real *gradTfms_data      = THTensor_(data)(gradTfms);
    real *gradTfmpoints_data = THTensor_(data)(gradTfmpoints);
    real *tfmpoints_data     = THTensor_(data)(tfmpoints);

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
                real x = *(points_data + 0*ps[1] + valp);
                real y = *(points_data + 1*ps[1] + valp);
                real z = *(points_data + 2*ps[1] + valp);

                // Get gradient w.r.t output point (gpt)
                real gxt = *(gradTfmpoints_data + 0*ps[1] + valp);
                real gyt = *(gradTfmpoints_data + 1*ps[1] + valp);
                real gzt = *(gradTfmpoints_data + 2*ps[1] + valp);

                // Gradients w.r.t pts, masks & tfms
                long valm = b*ms[0] + r*ms[2] + c*ms[3];
                real gx = 0, gy = 0, gz = 0; // Grads w.r.t input pts
                for (k = 0; k < nSE3; k++)
                {
                    // Get transform & wt
                    real w_k = *(masks_data + k*ms[1] + valm);   // Get the weight for the 'k'th transform "
                    real *T  = tfms_data + b*ts[0] + k*ts[1];     // Get the 'k'th transform

                    // === Gradient w.r.t input point (p = R^T * gpt, summed across all the "k" transforms)
                    gx += w_k * (T[0] * gxt + T[4] * gyt + T[8]  * gzt);
                    gy += w_k * (T[1] * gxt + T[5] * gyt + T[9]  * gzt);
                    gz += w_k * (T[2] * gxt + T[6] * gyt + T[10] * gzt);

                    // === Gradient w.r.t mask (w_k) = (R_k^T * p + t_k) * gpt
                    *(gradMasks_data + k*ms[1] + valm) = gxt * (T[0] * x + T[1] * y + T[2]  * z + T[3]) +
                                                         gyt * (T[4] * x + T[5] * y + T[6]  * z + T[7]) +
                                                         gzt * (T[8] * x + T[9] * y + T[10] * z + T[11]);

                    // === Gradients w.r.t transforms (t_k)
                    real *gT = gradTfms_data + b*ts[0] + k*ts[1]; // Get the gradient of the 'k'th transform

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
}

#endif

