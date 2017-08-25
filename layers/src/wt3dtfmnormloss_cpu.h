// == Float
float Weighted3DTransformNormLoss_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetflows,
			float normWt,
			int normPerPt,
            int sizeAverage);

void Weighted3DTransformNormLoss_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetflows,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
            THFloatTensor *gradOutput,
            float normWt,
            int normPerPt,
            int sizeAverage);

// == Double
double Weighted3DTransformNormLoss_forward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetflows,
			float normWt,
			int normPerPt,
            int sizeAverage);

void Weighted3DTransformNormLoss_backward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetflows,
			THDoubleTensor *gradPoints,
			THDoubleTensor *gradMasks,
			THDoubleTensor *gradTfms,
            THDoubleTensor *gradOutput,
            float normWt,
            int normPerPt,
            int sizeAverage);
