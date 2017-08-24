#ifndef _WT3DTFMLOSS_KERNEL
#define _WT3DTFMLOSS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int Weighted3DTransformLoss_ForwardLauncher(const float *points, const float *masks, const float *tfms, const float *targetpts,
                                            int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
                                            const long *ps, const long *ms, const long *ts,
                                            cudaStream_t stream);

int Weighted3DTransformLoss_BackwardLauncher(const float *points, const float *masks, const float *tfms, const float *targetpts,
                                             float *gradPoints, float *gradMasks, float *gradTfms, int useMaskGradMag,
                                             int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
                                             const long *ps, const long *ms, const long *ts,
                                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

