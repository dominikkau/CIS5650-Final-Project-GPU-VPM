#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>
#include "relaxation.h"
#include "common.h"
#include "kernels.h"
#include "sfs.h"
#include "particlebuffer.h"
#include "particlefield.h"

__host__ __device__ inline vpmvec3 xDotNablaY(const vpmvec3& x, const vpmmat3& jacobianY) {
#ifdef TRANSPOSED
    return jacobianY * x;
#else
    return x * jacobianY;
#endif
}

__host__ __device__ inline vpmvec3 nablaCrossX(const vpmmat3& jacobianX) {
    return vpmvec3{
        jacobianX[1][2] - jacobianX[2][1],
        jacobianX[2][0] - jacobianX[0][2],
        jacobianX[0][1] - jacobianX[1][0]
    };
}

template <typename K>
__global__ void calcEstrNaive(int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset = false, vpmfloat testFilterFactor = 1.0f);

void calcEstrNaiveWrapper(CUDAKernelParams params, int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, KernelType kernel, bool reset = false, vpmfloat testFilterFactor = 1.0f);

template <typename K>
__global__ void calcVelJacNaive(int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset = false, vpmfloat testFilterFactor = 1.0f);

void calcVelJacNaiveWrapper(CUDAKernelParams params, int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, KernelType kernel, bool reset = false, vpmfloat testFilterFactor = 1.0f);

__global__ void rungeKuttaStep(int N, ParticleBuffer particles, vpmfloat a, vpmfloat b, vpmfloat dt,
    vpmfloat zeta0, vpmvec3 Uinf);

void rungeKutta(ParticleField& field, vpmfloat dt, bool useRelax, int numBlocks, int blockSize, cudaStream_t stream = 0);

void runSimulation();

void runVPM(
    unsigned int maxParticles,
    unsigned int numParticles,
    unsigned int numTimeSteps,
    vpmfloat dt,
    unsigned int fileSaveSteps,
    vpmvec3 uInf,
    ParticleBuffer particleBuffer,
    RelaxationScheme *relaxation,
    SFSScheme *sfs,
    KernelType kernel,
    int blockSize,
    std::string filename
);