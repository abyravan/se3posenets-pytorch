// == Float
float Weighted3DTransformLoss_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetpoints,
            THFloatTensor *numpts);

void Weighted3DTransformLoss_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetpoints,
            THFloatTensor *numpts,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
            THFloatTensor *gradOutput,
            int useMaskGradMag);

// == Double
double Weighted3DTransformLoss_forward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetpoints,
            THDoubleTensor *numpts);

void Weighted3DTransformLoss_backward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetpoints,
            THDoubleTensor *numpts,
			THDoubleTensor *gradPoints,
			THDoubleTensor *gradMasks,
			THDoubleTensor *gradTfms,
            THDoubleTensor *gradOutput,
            int useMaskGradMag);
