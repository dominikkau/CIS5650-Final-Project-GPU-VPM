#pragma once

#include <cuda.h>
//#include <cuda_runtime.h>
#include "../vpmcore/common.h"

void testP2MKernel();

template <typename T>
__device__ void inline d_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__host__ __device__ int point_index_analytic(int index, int level, int point_count);
template<unsigned int threads_per_cell>
__global__ void fmmP2M(vpmvec4* xqs, int N, float* Rout, int p, int depth);