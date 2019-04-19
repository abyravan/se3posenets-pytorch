#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#define WARPSIZE 32

// Warp-shuffle to compute the sum across the warp very efficiently
__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

/// Get the (batch,row,col) indices corresponding to a given thread index (3D point index)
__device__ void getCoordinates(const int tid, const int nrows, const int ncols,
                               int &batch, int &row, int &col)
{
    // Get col id
    int id = tid;
    col = id % ncols;
    id = id / ncols;

    // Get row id
    row = id % nrows;
    id = id / nrows;

    // Get batch id
    batch = id;
}

#endif
