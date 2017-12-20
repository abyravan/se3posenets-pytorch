// == Float
int ComputeNormals_float(
            THFloatTensor *cloud_1,
            THFloatTensor *cloud_2,
            THByteTensor  *label_1,
            THFloatTensor *deltaposes_12,
            THFloatTensor *normals_1,
            THFloatTensor *tnormals_2,
            float maxdepthdiff);

int BilateralDepthSmoothing_float(
            THFloatTensor *depth_i,
            THFloatTensor *depth_o,
            THFloatTensor *lochalfkernel,
            float depthstd);
