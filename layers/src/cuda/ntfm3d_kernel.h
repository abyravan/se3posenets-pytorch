#ifndef _NTFM3D_KERNEL
#define _NTFM3D_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int NTfm3D_ForwardLauncher(const float *points, const float *masks, const float *tfms, float *tfmpoints,
								  int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
								  const long *ps, const long *ms, const long *ts,
								  cudaStream_t stream);

int NTfm3D_BackwardLauncher(const float *points, const float *masks, const float *tfms, const float *tfmpoints,
									float *gradPoints, float *gradMasks, float *gradTfms, const float *gradTfmPoints,
									int useMaskGradMag, int batchSize, int ndim, int nrows, int ncols, int nSE3, int nTfmParams,
									const long *ps, const long *ms, const long *ts,
									cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

