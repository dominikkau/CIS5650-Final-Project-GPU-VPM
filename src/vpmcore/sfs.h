#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

class ParticleField;
class ParticleBuffer;

struct SFSScheme {
    virtual void operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream = 0) = 0;
};

class DynamicSFS : public SFSScheme {
private:
    vpm::real minC;
    vpm::real maxC;
    vpm::real alpha;
    vpm::real relaxFactor; // relaxation factor for Lagrangian average
    bool forcePositive;

public:
    DynamicSFS(vpm::real minC = 0, vpm::real maxC = 1, vpm::real alpha = 0.667, vpm::real relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    void operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void calculateTemporary(vpm::pidx_t N, vpm::mat3* __restrict__ M, const vpm::mat3* __restrict__ J, const vpm::vec3* __restrict__ Gamma,
    const vpm::vec3* __restrict__ SFS, bool testFilter);

__global__ void calculateCoefficient(vpm::pidx_t N, const vpm::mat3* __restrict__ M, const vpm::vec3* __restrict__ Gamma,
    const vpm::vec3* __restrict__ SFS, const vpm::real* __restrict__ sigma, vpm::vec3* __restrict__ C, vpm::real zeta0,
    vpm::real alpha, vpm::real relaxFactor, bool forcePositive, vpm::real minC, vpm::real maxC);

class NoSFS : public SFSScheme {
    void operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};