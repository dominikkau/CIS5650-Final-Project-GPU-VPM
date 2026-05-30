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
    vpm::vec2 limC;
    vpm::real alpha;
    vpm::real relaxFactor; // relaxation factor for Lagrangian average
    bool forcePositive;

public:
    DynamicSFS(vpm::vec2 limC = vpm::vec2{ 0.0f, 1.0f }, vpm::real alpha = 0.667, vpm::real relaxFactor = 0.005, bool forcePositive = true)
        : limC(limC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    void operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

template <bool testFilter>
__global__ void calculateTemporary(vpm::pidx_t N, vpm::mat3* __restrict__ M, const vpm::mat3* __restrict__ J, const vpm::real* __restrict__ GammaX,
    const vpm::real* __restrict__ GammaY, const vpm::real* __restrict__ GammaZ, const vpm::vec3* __restrict__ SFS);

__global__ void calculateCoefficient(vpm::pidx_t N, const vpm::mat3* __restrict__ M, const vpm::real* __restrict__ GammaX,
    const vpm::real* __restrict__ GammaY, const vpm::real* __restrict__ GammaZ, const vpm::vec3* __restrict__ SFS, const vpm::real* __restrict__ sigma, vpm::vec3* __restrict__ C, vpm::real zeta0,
    vpm::real alpha, vpm::real relaxFactor, bool forcePositive, vpm::vec2 limC);

class NoSFS : public SFSScheme {
    void operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};