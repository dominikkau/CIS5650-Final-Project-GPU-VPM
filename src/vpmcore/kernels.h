#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

struct Kernel {
    virtual __host__ __device__ inline vpmfloat zeta(vpmfloat r) = 0;
    virtual __host__ __device__ inline vpmfloat g(vpmfloat r) = 0;
    virtual __host__ __device__ inline vpmfloat dgdr(vpmfloat r) = 0;
    virtual __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) = 0;
}

struct SingularKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) { return (r == 0.0f) ? 1.0f : 0.0f; }
    __host__ __device__ inline vpmfloat g(vpmfloat r) { return 1.0f; }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) { return 0.0f; }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) { return vpmvec2{ 1.0f, 0.0f }; }
};

struct GaussianKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) {
        return const3 * exp(-r * r * r);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) {
        return 1.0f - exp(-r * r * r);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) {
        return 3.0f * r * r * exp(-r * r * r);
    }
    __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) {
        const vpmfloat tmp = exp(-r * r * r);
        return vpmvec2{ 1.0f - tmp, 3.0f * r * r * tmp };
    }
};

struct GaussianErfKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) {
        return const1 * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) {
        return erf(r / sqrt2) - const2 * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) {
        return const2 * r * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) {
        const vpmfloat tmp = const2 * r * exp(-r * r / 2.0f);
        return vpmvec2{ erf(r / sqrt2) - tmp, r * tmp };
    }
};

struct WinckelmansKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) {
        return const4 * 7.5f / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) {
        return r * r * r * (r * r + 2.5f) / pow(r * r + 1.0f, 2.5f);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) {
        return 7.5f * r * r / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) {
        const vpmfloat tmp = pow(r * r + 1.0f, 2.5f);
        return vpmvec2{ r * r * r * (r * r + 2.5f) / tmp,
                          7.5f * r * r / (tmp * (r * r + 1.0f)) };
    }
};