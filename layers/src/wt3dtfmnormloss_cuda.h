float Weighted3DTransformNormLoss_forward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetflows,
            THCudaTensor *numpts,
            float normWt,
            int normPerPt);

void Weighted3DTransformNormLoss_backward_cuda(
			THCudaTensor *points,
			THCudaTensor *masks,
			THCudaTensor *tfms,
			THCudaTensor *targetflows,
            THCudaTensor *numpts,
			THCudaTensor *gradPoints,
			THCudaTensor *gradMasks,
			THCudaTensor *gradTfms,
            THCudaTensor *gradOutput,
            float normWt,
            int normPerPt);
