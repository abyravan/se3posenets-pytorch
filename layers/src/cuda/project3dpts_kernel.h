#ifndef _PROJECT3DPTS_KERNEL
#define _PROJECT3DPTS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int Project3DPointsToSubPixelDepth_ForwardLauncher(const float *input, float *indexMap, float *output,
                                                   int batchSize, int nrows, int ncols,
                                                   float fx, float fy, float cx, float cy,
                                                   const long *is, const long *iMs,
                                                   cudaStream_t stream);

int Project3DPointsToSubPixelDepth_BackwardLauncher(const float *input, const float *indexMap,
                                                    float *gradInput, const float *gradOutput,
                                                    int batchSize, int nrows, int ncols,
                                                    float fx, float fy, float cx, float cy,
                                                    const long *is, const long *iMs,
                                                    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif