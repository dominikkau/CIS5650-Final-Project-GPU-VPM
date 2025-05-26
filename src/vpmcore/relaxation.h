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
    vpmfloat relaxFactor;

public:
    PedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}

    void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void pedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

class CorrectedPedrizzettiRelaxation : public RelaxationScheme {
private:
    vpmfloat relaxFactor;

public:
    CorrectedPedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}

    void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void correctedPedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

class NoRelaxation : public RelaxationScheme {
    inline void operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0) {}
};