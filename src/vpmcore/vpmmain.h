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

__host__ __device__ inline vpm::vec3 xDotNablaY(const vpm::vec3& x, const vpm::mat3& jacobianY) {
#ifdef TRANSPOSED
    return jacobianY * x;
#else
    return x * jacobianY;
#endif
}

__host__ __device__ inline vpm::vec3 nablaCrossX(const vpm::mat3& jacobianX) {
    return vpm::vec3{
        jacobianX[1][2] - jacobianX[2][1],
        jacobianX[2][0] - jacobianX[0][2],
        jacobianX[0][1] - jacobianX[1][0]
    };
}

template <typename K>
__global__ void calcEstrNaive(vpm::pidx_t targetN, vpm::pidx_t sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset = false, vpm::real testFilterFactor = 1.0f);

void calcEstrNaiveWrapper(CUDAKernelParams params, vpm::pidx_t targetN, vpm::pidx_t sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, KernelType kernel, bool reset = false, vpm::real testFilterFactor = 1.0f);

template <typename K>
__global__ void calcVelJacNaive(vpm::pidx_t targetN, vpm::pidx_t sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset = false, vpm::real testFilterFactor = 1.0f);

void calcVelJacNaiveWrapper(CUDAKernelParams params, vpm::pidx_t targetN, vpm::pidx_t sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, KernelType kernel, bool reset = false, vpm::real testFilterFactor = 1.0f);

__global__ void rungeKuttaStep(vpm::pidx_t N, ParticleBuffer particles, vpm::real a, vpm::real b, vpm::real dt,
    vpm::real zeta0, vpm::vec3 Uinf);

void rungeKutta(ParticleField& field, vpm::real dt, bool useRelax, int numBlocks, int blockSize, cudaStream_t stream = 0);

void writeVTK(ParticleBuffer& particles, vpm::pidx_t N, const std::string& filename, int outputMask);

void runSimulation();

void runVPM(
    vpm::pidx_t maxParticles,
    vpm::pidx_t numParticles,
    unsigned int numTimeSteps,
    vpm::real dt,
    unsigned int fileSaveSteps,
    vpm::vec3 uInf,
    ParticleBuffer particleBuffer,
    RelaxationScheme *relaxation,
    SFSScheme *sfs,
    KernelType kernel,
    int blockSize,
    std::string filename
);