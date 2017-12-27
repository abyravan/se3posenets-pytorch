#ifndef _WT3DTFMLOSS_KERNEL
#define _WT3DTFMLOSS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

float Weighted3DTransformNormLoss_ForwardLauncher(const float *points, const float *masks, const float *tfms, const float *targetflows, const float *numpts,
                                            int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
                                            float normWt, int normPerPt,
                                            const long *ps, const long *ms, const long *ts,
                                            cudaStream_t stream);

void Weighted3DTransformNormLoss_BackwardLauncher(const float *points, const float *masks, const float *tfms, const float *targetflows, const float *numpts,
                                             float *gradPoints, float *gradMasks, float *gradTfms,
                                             int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
                                             float normWt, int normPerPt,
                                             const long *ps, const long *ms, const long *ts,
                                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

