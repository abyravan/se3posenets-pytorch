// == Float
float Weighted3DTransformNormLoss_forward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetflows,
            THFloatTensor *numpts,
			float normWt,
			int normPerPt);

void Weighted3DTransformNormLoss_backward_float(
			THFloatTensor *points,
			THFloatTensor *masks,
			THFloatTensor *tfms,
			THFloatTensor *targetflows,
            THFloatTensor *numpts,
			THFloatTensor *gradPoints,
			THFloatTensor *gradMasks,
			THFloatTensor *gradTfms,
            THFloatTensor *gradOutput,
            float normWt,
            int normPerPt);

// == Double
double Weighted3DTransformNormLoss_forward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetflows,
            THDoubleTensor *numpts,
			float normWt,
			int normPerPt);

void Weighted3DTransformNormLoss_backward_double(
			THDoubleTensor *points,
			THDoubleTensor *masks,
			THDoubleTensor *tfms,
			THDoubleTensor *targetflows,
            THDoubleTensor *numpts,
			THDoubleTensor *gradPoints,
			THDoubleTensor *gradMasks,
			THDoubleTensor *gradTfms,
            THDoubleTensor *gradOutput,
            float normWt,
            int normPerPt);
