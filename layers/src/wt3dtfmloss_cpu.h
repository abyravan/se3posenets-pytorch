// == Float
float Weighted3DTransformLoss_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetpoints,
            int sizeAverage);

void Weighted3DTransformLoss_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetpoints,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
            THFloatTensor *gradOutput,
            int sizeAverage);

// == Double
double Weighted3DTransformLoss_forward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetpoints,
            int sizeAverage);

void Weighted3DTransformLoss_backward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetpoints,
			THDoubleTensor *gradPoints,
			THDoubleTensor *gradMasks,
			THDoubleTensor *gradTfms,
            THDoubleTensor *gradOutput,
            int sizeAverage);
