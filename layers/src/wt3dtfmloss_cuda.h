float Weighted3DTransformLoss_forward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetpoints,
			int sizeAverage);

void Weighted3DTransformLoss_backward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetpoints,
			THCudaTensor *gradPoints,
			THCudaTensor *gradMasks,
			THCudaTensor *gradTfms,
            THCudaTensor *gradOutput,
			int sizeAverage,
            int useMaskGradMag);
