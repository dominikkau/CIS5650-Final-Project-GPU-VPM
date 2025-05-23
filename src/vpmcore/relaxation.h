#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

struct ParticleField;

struct RelaxationScheme {
    void operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

class PedrizzettiRelaxation {
private:
    vpmfloat relaxFactor;

    static __global__ void relax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

public:
    PedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}

    void operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void pedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

class CorrectedPedrizzettiRelaxation {
private:
    vpmfloat relaxFactor;

    static __global__ void relax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

public:
    CorrectedPedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}

    void operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void correctedPedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

struct NoRelaxation {
    inline void operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0) {}
};