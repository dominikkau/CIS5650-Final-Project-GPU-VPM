#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../vpmcore/common.h"

void testP2MKernel();

template <typename T>
__device__ void inline d_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__host__ __device__ int point_index_analytic(int index, int level, int point_count);
__device__ void addToM(float* buffer, float q, float* M, int p, int n);
__global__ void fmm_P2M(vpmvec4* xqs, int N, float* Rout, int p, int depth);