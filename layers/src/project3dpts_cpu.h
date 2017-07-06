// == Float
int Project3DPointsToSubPixelDepth_forward_float(
                                                 THFloatTensor *input,
                                                 THFloatTensor *indexMap,
                                                 THFloatTensor *output,
                                                 float fy, float fx,
                                                 float cy, float cx);

int Project3DPointsToSubPixelDepth_backward_float(
                                                  THFloatTensor *input,
                                                  THFloatTensor *indexMap,
                                                  THFloatTensor *gradInput,
                                                  THFloatTensor *gradOutput,
                                                  float fy, float fx,
                                                  float cy, float cx);

// == Double
int Project3DPointsToSubPixelDepth_forward_double(
                                                  THDoubleTensor *input,
                                                  THDoubleTensor *indexMap,
                                                  THDoubleTensor *output,
                                                  double fy, double fx,
                                                  double cy, double cx);

int Project3DPointsToSubPixelDepth_backward_double(
                                                   THDoubleTensor *input,
                                                   THDoubleTensor *indexMap,
                                                   THDoubleTensor *gradInput,
                                                   THDoubleTensor *gradOutput,
                                                   double fy, double fx,
                                                   double cy, double cx);