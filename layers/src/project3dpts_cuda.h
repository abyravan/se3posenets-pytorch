int Project3DPointsToSubPixelDepth_forward_cuda(
                                                THCudaTensor *input,
                                                THCudaTensor *indexMap,
                                                THCudaTensor *output,
                                                float fy, float fx,
                                                float cy, float cx);

int Project3DPointsToSubPixelDepth_backward_cuda(
                                                 THCudaTensor *input,
                                                 THCudaTensor *indexMap,
                                                 THCudaTensor *gradInput,
                                                 THCudaTensor *gradOutput,
                                                 float fy, float fx,
                                                 float cy, float cx);