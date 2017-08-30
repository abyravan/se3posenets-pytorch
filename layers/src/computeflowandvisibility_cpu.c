#include <TH/TH.h>
#include <assert.h>
#include <math.h>

void compute_visibility(
        const float *local1,
        const float *local2,
        const unsigned char *label1,
        const unsigned char *label2,
        const float *poses2,
        unsigned char *visible1,
        const long *cs,
        const long *ls,
        const long *ps,
        float fx,
        float fy,
        float cx,
        float cy,
        float threshold,
        float winsize,
        long batchsize,
        long nrows,
        long ncols)
{
    // Setup extra params
    float sqthresh   = pow(threshold,2); // Threshold on squared distance
    int winhalfsize  = floor(winsize/2.0); // -winhalfsize -> (-winhalfsize + winsize-1)

    // Project to get pixel in target image, check directly instead of going through a full projection step where we check for visibility
    // Iterate over the images and compute the data-associations (using the vertex map images)
    long b,r,c;
    for(b = 0; b < batchsize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                // Get local pt
                long valc = b*cs[0] + r*cs[2] + c*cs[3]; // Don't add stride along 3D dim
                float xi = *(local1 + 0*cs[1] + valc);
                float yi = *(local1 + 1*cs[1] + valc);
                float zi = *(local1 + 2*cs[1] + valc);

                // Get link label for that input point
                unsigned char mi = *(label1 + b*ls[0] + r*ls[2] + c*ls[3]);

                // In case the ID is background, then skip DA (we need to check for z < 0 => this is local frame of reference, not camera)
                if (mi == 0)
                {
                    *(visible1 + b*ls[0] + r*ls[2] + c*ls[3]) = 1; // Assume that BG points are always visible
                    continue;
                }

                // Find the 3D point where this vertex projects onto in the target frame
                const float *T  = poses2 + b*ps[0] + mi*ps[1]; // Get the 'mi'th transform
                float xp  = T[0] * xi + T[1] * yi + T[2]  * zi + T[3];
                float yp  = T[4] * xi + T[5] * yi + T[6]  * zi + T[7];
                float zp  = T[8] * xi + T[9] * yi + T[10] * zi + T[11];

                // Project target 3D point (in cam frame) onto canvas to get approx pixel location to search for DA
                int cpix = (int) round((xp/zp)*fx + cx);
                int rpix = (int) round((yp/zp)*fy + cy);
                if (rpix < 0 || rpix >= nrows || cpix < 0 || cpix >= ncols) continue;

                // Check in a region around this point to see if you can find a match in the local vertices of frame t2
                float mindist = HUGE_VALF;
                int mintr = -1, mintc = -1;
                int tr, tc;
                for (tr = (rpix-winhalfsize); tr < (rpix-winhalfsize+winsize); tr++)
                {
                    for (tc = (cpix-winhalfsize); tc < (cpix-winhalfsize+winsize); tc++)
                    {
                        // Check limits
                        if (tr < 0 || tr >= nrows || tc < 0 || tc >= ncols) continue;

                        // Get target value
                        long valtc = b*cs[0] + tr*cs[2] + tc*cs[3]; // Don't add stride along 3D dim
                        float xt = *(local2 + 0*cs[1] + valtc);
                        float yt = *(local2 + 1*cs[1] + valtc);
                        float zt = *(local2 + 2*cs[1] + valtc);

                        // Get link label for that target point
                        unsigned char mt = *(label2 + b*ls[0] + tr*ls[2] + tc*ls[3]);

                        // Compare only in the same mesh, if not continue
                        if (mt != mi) continue;

                        // Now check distance in local-coordinates
                        // If this is closer than previous NN & also less than the outlier threshold, count for loss
                        float dist = pow(xi-xt, 2) + pow(yi-yt, 2) + pow(zi-zt, 2);
                        if ((dist < mindist) && (dist < sqthresh))
                        {
                            mindist = dist;
                            mintr = tr;
                            mintc = tc;
                        }
                    }
                }

                // In case we found a match, update outputs
                if(mintr != -1 && mintc != -1)
                {
                    *(visible1 + b*ls[0] + r*ls[2] + c*ls[3]) = 1; // visible
                }
            }
        }
    }
}

// ===== FLOAT DATA

int ComputeFlowAndVisibility_float(
            THFloatTensor *cloud_1,
            THFloatTensor *cloud_2,
            THByteTensor  *label_1,
            THByteTensor  *label_2,
            THFloatTensor *poses_1,
            THFloatTensor *poses_2,
            THFloatTensor *poseinvs_1,
            THFloatTensor *poseinvs_2,
            THFloatTensor *fwdflows,
            THFloatTensor *bwdflows,
            THByteTensor  *fwdvisibility,
            THByteTensor  *bwdvisibility,
            float fx,
            float fy,
            float cx,
            float cy,
            float threshold,
            float winsize)
{
    // Initialize vars
    long batchsize = cloud_1->size[0];
    long ndim      = cloud_1->size[1];
    long nrows     = cloud_1->size[2];
    long ncols     = cloud_1->size[3];
    assert(ndim == 3);

    // New memory in case the inputs are not contiguous
    cloud_1 = THFloatTensor_newContiguous(cloud_1);
    cloud_2 = THFloatTensor_newContiguous(cloud_2);
    label_1 = THByteTensor_newContiguous(label_1);
    label_2 = THByteTensor_newContiguous(label_2);
    poses_1 = THFloatTensor_newContiguous(poses_1);
    poses_2 = THFloatTensor_newContiguous(poses_2);
    poseinvs_1 = THFloatTensor_newContiguous(poseinvs_1);
    poseinvs_2 = THFloatTensor_newContiguous(poseinvs_2);
    fwdflows      = THFloatTensor_newContiguous(fwdflows);
    bwdflows      = THFloatTensor_newContiguous(bwdflows);
    fwdvisibility = THByteTensor_newContiguous(fwdvisibility);
    bwdvisibility = THByteTensor_newContiguous(bwdvisibility);

    // Get data pointers
    const float *cloud1_data 	     = THFloatTensor_data(cloud_1);
    const float *cloud2_data 	     = THFloatTensor_data(cloud_2);
    const unsigned char *label1_data = THByteTensor_data(label_1);
    const unsigned char *label2_data = THByteTensor_data(label_2);
    const float *poses1_data         = THFloatTensor_data(poses_1);
    const float *poses2_data 	     = THFloatTensor_data(poses_2);
    const float *poseinvs1_data 	 = THFloatTensor_data(poseinvs_1);
    const float *poseinvs2_data 	 = THFloatTensor_data(poseinvs_2);
    float *fwdflows_data             = THFloatTensor_data(fwdflows);
    float *bwdflows_data             = THFloatTensor_data(bwdflows);
    unsigned char *fwdvisibility_data = THByteTensor_data(fwdvisibility);
    unsigned char *bwdvisibility_data = THByteTensor_data(bwdvisibility);

    // Set visibility to zero by default
    THByteTensor_fill(fwdvisibility, 0);
    THByteTensor_fill(bwdvisibility, 0);

    // Get strides
    long *cs = cloud_1->stride;
    long *ls = label_1->stride;
    long *ps = poses_1->stride;

    /// ====== Iterate over all points, compute local coordinates
    long b,r,c;
    for(b = 0; b < batchsize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                /// === Compute local co-ordinate @ t, save in flow for now
                // Get cam pt @ t
                long valc = b*cs[0] + r*cs[2] + c*cs[3]; // Don't add stride along 3D dim
                float x1 = *(cloud1_data + 0*cs[1] + valc);
                float y1 = *(cloud1_data + 1*cs[1] + valc);
                float z1 = *(cloud1_data + 2*cs[1] + valc);

                // Get transform for link of that point
                unsigned char l1 = *(label1_data + b*ls[0] + r*ls[2] + c*ls[3]);
                const float *T1  = poseinvs1_data + b*ps[0] + l1*ps[1]; // Get the 'l1'th transform

                // Transform to local frame; local_pt = T_cam_to_local * global_pt
                *(fwdflows_data + 0*cs[1] + valc) = T1[0] * x1 + T1[1] * y1 + T1[2]  * z1 + T1[3];
                *(fwdflows_data + 1*cs[1] + valc) = T1[4] * x1 + T1[5] * y1 + T1[6]  * z1 + T1[7];
                *(fwdflows_data + 2*cs[1] + valc) = T1[8] * x1 + T1[9] * y1 + T1[10] * z1 + T1[11];

                /// === Compute local co-ordinate @ t+1, save in flow for now
                // Get cam pt @ t+1
                float x2 = *(cloud2_data + 0*cs[1] + valc);
                float y2 = *(cloud2_data + 1*cs[1] + valc);
                float z2 = *(cloud2_data + 2*cs[1] + valc);

                // Get transform for link of that point
                unsigned char l2 = *(label2_data + b*ls[0] + r*ls[2] + c*ls[3]);
                const float *T2  = poseinvs2_data + b*ps[0] + l2*ps[1]; // Get the 'l2'th transform

                // Transform to local frame; local_pt = T_cam_to_local * global_pt
                *(bwdflows_data + 0*cs[1] + valc) = T2[0] * x2 + T2[1] * y2 + T2[2]  * z2 + T2[3];
                *(bwdflows_data + 1*cs[1] + valc) = T2[4] * x2 + T2[5] * y2 + T2[6]  * z2 + T2[7];
                *(bwdflows_data + 2*cs[1] + valc) = T2[8] * x2 + T2[9] * y2 + T2[10] * z2 + T2[11];
            }
        }
    }

    /// ======== Compute visibility masks
    // t -> t+1
    compute_visibility(fwdflows_data, bwdflows_data, label1_data, label2_data, poses2_data,
                       fwdvisibility_data, cs, ls, ps,
                       fx, fy, cx, cy, threshold, winsize,
                       batchsize, nrows, ncols);

    // t+1 -> t (TODO: This can be efficient, as most of this is pre-computed in previous step)
    compute_visibility(bwdflows_data, fwdflows_data, label2_data, label1_data, poses1_data,
                       bwdvisibility_data, cs, ls, ps,
                       fx, fy, cx, cy, threshold, winsize,
                       batchsize, nrows, ncols);

    /// ======== Compute flows
    for(b = 0; b < batchsize; b++)
    {
        for(r = 0; r < nrows; r++)
        {
            for(c = 0; c < ncols; c++)
            {
                /// == Flow from t-> t+1
                // Get local point
                long valc = b*cs[0] + r*cs[2] + c*cs[3]; // Don't add stride along 3D dim
                float x1 = *(fwdflows_data + 0*cs[1] + valc);
                float y1 = *(fwdflows_data + 1*cs[1] + valc);
                float z1 = *(fwdflows_data + 2*cs[1] + valc);

                // Get link label for that input point
                unsigned char m1 = *(label1_data + b*ls[0] + r*ls[2] + c*ls[3]);

                // In case the ID is background, then skip DA (we need to check for z < 0 => this is local frame of reference, not camera)
                if (m1 == 0)
                {
                    *(fwdflows_data + 0*cs[1] + valc) = 0; // flow = 0 for BG point
                    *(fwdflows_data + 1*cs[1] + valc) = 0; // flow = 0 for BG point
                    *(fwdflows_data + 2*cs[1] + valc) = 0; // flow = 0 for BG point
                }
                else
                {
                    // Find the 3D point where this vertex projects onto in the target frame
                    const float *T  = poses2_data + b*ps[0] + m1*ps[1]; // Get the 'mi'th transform
                    float x1p  = T[0] * x1 + T[1] * y1 + T[2]  * z1 + T[3];
                    float y1p  = T[4] * x1 + T[5] * y1 + T[6]  * z1 + T[7];
                    float z1p  = T[8] * x1 + T[9] * y1 + T[10] * z1 + T[11];

                    // Flow is difference between this pt and input point
                    *(fwdflows_data + 0*cs[1] + valc) = (x1p - *(cloud1_data + 0*cs[1] + valc));
                    *(fwdflows_data + 1*cs[1] + valc) = (y1p - *(cloud1_data + 1*cs[1] + valc));
                    *(fwdflows_data + 2*cs[1] + valc) = (z1p - *(cloud1_data + 2*cs[1] + valc));
                }

                /// == Flow from t+1 -> t
                // Get local point
                float x2 = *(bwdflows_data + 0*cs[1] + valc);
                float y2 = *(bwdflows_data + 1*cs[1] + valc);
                float z2 = *(bwdflows_data + 2*cs[1] + valc);

                // Get link label for that point
                unsigned char m2 = *(label2_data + b*ls[0] + r*ls[2] + c*ls[3]);

                // Compute flow
                if (m2 == 0)
                {
                    *(bwdflows_data + 0*cs[1] + valc) = 0; // flow = 0 for BG point
                    *(bwdflows_data + 1*cs[1] + valc) = 0; // flow = 0 for BG point
                    *(bwdflows_data + 2*cs[1] + valc) = 0; // flow = 0 for BG point
                }
                else
                {
                    // Find the 3D point where this vertex projects onto in the target frame
                    const float *T  = poses1_data + b*ps[0] + m2*ps[1]; // Get the 'mi'th transform
                    float x2p  = T[0] * x2 + T[1] * y2 + T[2]  * z2 + T[3];
                    float y2p  = T[4] * x2 + T[5] * y2 + T[6]  * z2 + T[7];
                    float z2p  = T[8] * x2 + T[9] * y2 + T[10] * z2 + T[11];

                    // Flow is difference between this pt and input point
                    *(bwdflows_data + 0*cs[1] + valc) = (x2p - *(cloud2_data + 0*cs[1] + valc));
                    *(bwdflows_data + 1*cs[1] + valc) = (y2p - *(cloud2_data + 1*cs[1] + valc));
                    *(bwdflows_data + 2*cs[1] + valc) = (z2p - *(cloud2_data + 2*cs[1] + valc));
                }
            }
        }
    }

    /// ========= Free created memory
    THFloatTensor_free(cloud_1);
    THFloatTensor_free(cloud_2);
    THByteTensor_free(label_1);
    THByteTensor_free(label_2);
    THFloatTensor_free(poses_1);
    THFloatTensor_free(poses_2);
    THFloatTensor_free(fwdflows);
    THFloatTensor_free(bwdflows);
    THByteTensor_free(fwdvisibility);
    THByteTensor_free(bwdvisibility);

    // Return
    return 1;
}
