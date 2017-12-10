// == Float
int ComputeNormals_float(
            THFloatTensor *cloud_1,
            THFloatTensor *cloud_2,
            THByteTensor  *label_1,
            THFloatTensor *deltaposes_12,
            THFloatTensor *normals_1,
            THFloatTensor *tnormals_2,
            float maxdepthdiff);
