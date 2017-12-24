float Weighted3DTransformLoss_forward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetpoints,
			THCudaTensor *numpts);

void Weighted3DTransformLoss_backward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetpoints,
			THCudaTensor *numpts,
			THCudaTensor *gradPoints,
			THCudaTensor *gradMasks,
			THCudaTensor *gradTfms,
            THCudaTensor *gradOutput,
            int useMaskGradMag);
