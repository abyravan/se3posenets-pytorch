#include <TH/TH.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

bool check_limits(const unsigned int r, const unsigned int c, const unsigned int maxr, const unsigned int maxc)
{
	return ((r >= 0) && (r < maxr) && (c >= 0) && (c <= maxc));
}

void compute_visibility_and_flows(
        const float *cloud1,
        const float *cloud2,
        const float *local1,
        const float *local2,
        const unsigned char *label1,
        const unsigned char *label2,
        const float *poses2,
        unsigned char *visible1,
        float *flows12,
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
    // Project to get pixel in target image, check directly instead of going through a full projection step where we check for visibility
    // Iterate over the images and compute the data-associations (using the vertex map images)
    long b,r,c;
    for(b = 0; b < batchsize; b++)
    {
        // Iterate over the depth image & compute flows
        const float *depth2 = cloud2 + b*cs[0] + 2*cs[1];
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
                float csubpix = (xp/zp)*fx + cx;
                float rsubpix = (yp/zp)*fy + cy;

                // Get the target depth & compute corresponding 3D point in the target
                const unsigned int c1 = floor(csubpix);
                const unsigned int c2 = c1 + 1;
                const unsigned int r1 = floor(rsubpix);
                const unsigned int r2 = r1 + 1;

                // Compute interpolated weights & depth
                float w = 0, z = 0;
					 int ct = 0;

                // 1,1
                if (check_limits(r1, c1, nrows, ncols) && (fabsf(zp - depth2[r1*cs[2] + c1*cs[3]]) < threshold))
                {
                    float wt = (r2-rsubpix) * (c2-csubpix);
                    w += wt;
                    z += wt * depth2[r1*cs[2] + c1*cs[3]];
						  ct++;
                }

                // 1,2
                if (check_limits(r2, c1, nrows, ncols) && (fabsf(zp - depth2[r2*cs[2] + c1*cs[3]]) < threshold))
                {
                    float wt = (rsubpix-r1) * (c2-csubpix);
                    w += wt;
                    z += wt * depth2[r2*cs[2] + c1*cs[3]];
						  ct++;
                }

                // 2,1
                if (check_limits(r1, c2, nrows, ncols) && (fabsf(zp - depth2[r1*cs[2] + c2*cs[3]]) < threshold))
                {
                    float wt = (r2-rsubpix) * (csubpix-c1);
                    w += wt;
                    z += wt * depth2[r1*cs[2] + c2*cs[3]];
						  ct++;
                }

                // 2,2
                if (check_limits(r2, c2, nrows, ncols) && (fabsf(zp - depth2[r2*cs[2] + c2*cs[3]]) < threshold))
                {
                    float wt = (rsubpix-r1) * (csubpix-c1);
                    w += wt;
                    z += wt * depth2[r2*cs[2] + c2*cs[3]];
						  ct++;
                }

                // Compute interpolated depth, flows & visibility
                // In case none of the points are within the threshold, then w = 0 here
                if (w > 0) //&& ct == 4)
                {
                    // == Divide by total weight to get interpolated depth
                    z /= w;
                    float x = ((csubpix - cx)/fx) * z;
                    float y = ((rsubpix - cy)/fy) * z;

                    // == Set visibility to true
                    *(visible1 + b*ls[0] + r*ls[2] + c*ls[3]) = 1; // visible

                    // == Flow is difference between that point @ t1 & DA point @ t2
                    // Point @ t1
                    float x1 = *(cloud1 + 0*cs[1] + valc);
                    float y1 = *(cloud1 + 1*cs[1] + valc);
                    float z1 = *(cloud1 + 2*cs[1] + valc);

                    // Flow from t1-t2
                    *(flows12 + 0*cs[1] + valc) = xp - x1;
                    *(flows12 + 1*cs[1] + valc) = yp - y1;
                    *(flows12 + 2*cs[1] + valc) = zp - z1;
                }
            }
        }
    }
}

// ===== FLOAT DATA

int ComputeFlowAndVisibility_Pts_float(
            THFloatTensor *cloud_1,
            THFloatTensor *cloud_2,
            THFloatTensor *local_1,
            THFloatTensor *local_2,
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

    // New memory in case the inputs are not contiguous (no need for the local stuff since its temp memory)
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
    local_1 = THFloatTensor_newContiguous(local_1);
    local_2 = THFloatTensor_newContiguous(local_2);

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
    float *local1_data                = THFloatTensor_data(local_1);
    float *local2_data                = THFloatTensor_data(local_2);

    // Set visibility to zero by default
    THByteTensor_fill(fwdvisibility, 0);
    THByteTensor_fill(bwdvisibility, 0);
    THFloatTensor_fill(fwdflows, 0);
    THFloatTensor_fill(bwdflows, 0);

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
                *(local1_data + 0*cs[1] + valc) = T1[0] * x1 + T1[1] * y1 + T1[2]  * z1 + T1[3];
                *(local1_data + 1*cs[1] + valc) = T1[4] * x1 + T1[5] * y1 + T1[6]  * z1 + T1[7];
                *(local1_data + 2*cs[1] + valc) = T1[8] * x1 + T1[9] * y1 + T1[10] * z1 + T1[11];

                /// === Compute local co-ordinate @ t+1, save in flow for now
                // Get cam pt @ t+1
                float x2 = *(cloud2_data + 0*cs[1] + valc);
                float y2 = *(cloud2_data + 1*cs[1] + valc);
                float z2 = *(cloud2_data + 2*cs[1] + valc);

                // Get transform for link of that point
                unsigned char l2 = *(label2_data + b*ls[0] + r*ls[2] + c*ls[3]);
                const float *T2  = poseinvs2_data + b*ps[0] + l2*ps[1]; // Get the 'l2'th transform

                // Transform to local frame; local_pt = T_cam_to_local * global_pt
                *(local2_data + 0*cs[1] + valc) = T2[0] * x2 + T2[1] * y2 + T2[2]  * z2 + T2[3];
                *(local2_data + 1*cs[1] + valc) = T2[4] * x2 + T2[5] * y2 + T2[6]  * z2 + T2[7];
                *(local2_data + 2*cs[1] + valc) = T2[8] * x2 + T2[9] * y2 + T2[10] * z2 + T2[11];
            }
        }
    }

    /// ======== Compute visibility masks
    // t -> t+1
    compute_visibility_and_flows(cloud1_data, cloud2_data, local1_data, local2_data, label1_data, label2_data, poses2_data,
                                 fwdvisibility_data, fwdflows_data, cs, ls, ps,
                                 fx, fy, cx, cy, threshold, winsize,
                                 batchsize, nrows, ncols);

    // t+1 -> t
    compute_visibility_and_flows(cloud2_data, cloud1_data, local2_data, local1_data, label2_data, label1_data, poses1_data,
                                 bwdvisibility_data, bwdflows_data, cs, ls, ps,
                                 fx, fy, cx, cy, threshold, winsize,
                                 batchsize, nrows, ncols);

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
    THFloatTensor_free(local_1);
    THFloatTensor_free(local_2);

    // Return
    return 1;
}
