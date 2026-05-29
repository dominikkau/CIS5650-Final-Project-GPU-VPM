#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "particlebuffer.h"

class ParticleField;

struct RelaxationScheme {
    virtual void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0) = 0;
};

class PedrizzettiRelaxation : public RelaxationScheme {
private:
    vpm::real relaxFactor;

public:
    PedrizzettiRelaxation(vpm::real relaxFactor) : relaxFactor(relaxFactor) {}

    void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void pedrizzettiRelax(vpm::pidx_t N, const vpm::mat3* __restrict__ J, vpm::vec3* __restrict__ Gamma, vpm::real relaxFactor);

class CorrectedPedrizzettiRelaxation : public RelaxationScheme {
private:
    vpm::real relaxFactor;

public:
    CorrectedPedrizzettiRelaxation(vpm::real relaxFactor) : relaxFactor(relaxFactor) {}

    void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void correctedPedrizzettiRelax(vpm::pidx_t N, const vpm::mat3* __restrict__ J, vpm::vec3* __restrict__ Gamma, vpm::real relaxFactor);

class NoRelaxation : public RelaxationScheme {
    inline void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0) {}
};