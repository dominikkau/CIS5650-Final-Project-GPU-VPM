#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

enum class KernelType {
    SINGULAR,
    GAUSSIAN,
    GAUSSIAN_ERF,
    WINCKELMAN
};

struct Kernel {
    virtual __host__ __device__ inline vpmfloat zeta(vpmfloat r) const = 0;
    virtual __host__ __device__ inline vpmfloat g(vpmfloat r) const = 0;
    virtual __host__ __device__ inline vpmfloat dgdr(vpmfloat r) const = 0;
    virtual __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) const = 0;
};

struct SingularKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) const { return (r == 0.0f) ? 1.0f : 0.0f; }
    __host__ __device__ inline vpmfloat g(vpmfloat r) const { return 1.0f; }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) const { return 0.0f; }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) const { return vpmvec2{ 1.0f, 0.0f }; }
};

struct GaussianKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) const {
        return const3 * exp(-r * r * r);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) const {
        return 1.0f - exp(-r * r * r);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) const {
        return 3.0f * r * r * exp(-r * r * r);
    }
    __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) const {
        const vpmfloat tmp = exp(-r * r * r);
        return vpmvec2{ 1.0f - tmp, 3.0f * r * r * tmp };
    }
};

struct GaussianErfKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) const {
        return const1 * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) const {
        return erf(r / sqrt2) - const2 * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) const {
        return const2 * r * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) const {
        const vpmfloat tmp = const2 * r * exp(-r * r / 2.0f);
        return vpmvec2{ erf(r / sqrt2) - tmp, r * tmp };
    }
};

struct WinckelmansKernel : Kernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) const {
        return const4 * 7.5f / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) const {
        return r * r * r * (r * r + 2.5f) / pow(r * r + 1.0f, 2.5f);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) const {
        return 7.5f * r * r / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) const {
        const vpmfloat tmp = pow(r * r + 1.0f, 2.5f);
        return vpmvec2{ r * r * r * (r * r + 2.5f) / tmp,
                          7.5f * r * r / (tmp * (r * r + 1.0f)) };
    }
};

inline Kernel* getKernel(KernelType kernel)
{
    Kernel* kernelPointer{ nullptr };
    switch (kernel)
    {
    case KernelType::SINGULAR:
        kernelPointer = new SingularKernel();
        break;
    case KernelType::GAUSSIAN:
        kernelPointer = new GaussianKernel();
        break;
    case KernelType::GAUSSIAN_ERF:
        kernelPointer = new GaussianErfKernel();
        break;
    case KernelType::WINCKELMAN:
        kernelPointer = new WinckelmansKernel();
        break;
    default:
        // Default to GaussianKernel if unknown type
        kernelPointer = new GaussianKernel(); 
        break;
    }

    return kernelPointer;
}