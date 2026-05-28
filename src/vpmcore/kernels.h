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
    virtual __host__ __device__ inline vpm::real zeta(vpm::real r) const = 0;
    virtual __host__ __device__ inline vpm::real g(vpm::real r) const = 0;
    virtual __host__ __device__ inline vpm::real dgdr(vpm::real r) const = 0;
    virtual __host__ __device__ inline vpm::vec2 g_dgdr(vpm::real r) const = 0;
};

struct SingularKernel : Kernel {
    __host__ __device__ inline vpm::real zeta(vpm::real r) const { return (r == 0.0f) ? 1.0f : 0.0f; }
    __host__ __device__ inline vpm::real g(vpm::real r) const { return 1.0f; }
    __host__ __device__ inline vpm::real dgdr(vpm::real r) const { return 0.0f; }
    __host__ __device__ inline vpm::vec2  g_dgdr(vpm::real r) const { return vpm::vec2{ 1.0f, 0.0f }; }
};

struct GaussianKernel : Kernel {
    __host__ __device__ inline vpm::real zeta(vpm::real r) const {
        return const3 * exp(-r * r * r);
    }
    __host__ __device__ inline vpm::real g(vpm::real r) const {
        return 1.0f - exp(-r * r * r);
    }
    __host__ __device__ inline vpm::real dgdr(vpm::real r) const {
        return 3.0f * r * r * exp(-r * r * r);
    }
    __host__ __device__ inline vpm::vec2 g_dgdr(vpm::real r) const {
        const vpm::real tmp = exp(-r * r * r);
        return vpm::vec2{ 1.0f - tmp, 3.0f * r * r * tmp };
    }
};

struct GaussianErfKernel : Kernel {
    __host__ __device__ inline vpm::real zeta(vpm::real r) const {
        return const1 * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpm::real g(vpm::real r) const {
        return erf(r / sqrt2) - const2 * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpm::real dgdr(vpm::real r) const {
        return const2 * r * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpm::vec2  g_dgdr(vpm::real r) const {
        const vpm::real tmp = const2 * r * exp(-r * r / 2.0f);
        return vpm::vec2{ erf(r / sqrt2) - tmp, r * tmp };
    }
};

struct WinckelmansKernel : Kernel {
    __host__ __device__ inline vpm::real zeta(vpm::real r) const {
        return const4 * 7.5f / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpm::real g(vpm::real r) const {
        return r * r * r * (r * r + 2.5f) / pow(r * r + 1.0f, 2.5f);
    }
    __host__ __device__ inline vpm::real dgdr(vpm::real r) const {
        return 7.5f * r * r / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpm::vec2  g_dgdr(vpm::real r) const {
        const vpm::real tmp = pow(r * r + 1.0f, 2.5f);
        return vpm::vec2{ r * r * r * (r * r + 2.5f) / tmp,
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