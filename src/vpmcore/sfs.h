#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "particlebuffer.h"
#include "particlefield.h"

struct SFSScheme {
    virtual void operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream = 0) = 0;
};

class DynamicSFS : SFSScheme {
private:
    vpmfloat minC;
    vpmfloat maxC;
    vpmfloat alpha;
    vpmfloat relaxFactor; // relaxation factor for Lagrangian average
    bool forcePositive;

public:
    DynamicSFS(vpmfloat minC = 0, vpmfloat maxC = 1, vpmfloat alpha = 0.667, vpmfloat relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    void operator()(ParticleField& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};

__global__ void calculateTemporary(int N, ParticleBuffer particles, bool testFilter);

__global__ void calculateCoefficient(int N, ParticleBuffer particles, vpmfloat zeta0,
    vpmfloat alpha, vpmfloat relaxFactor, bool forcePositive, vpmfloat minC, vpmfloat maxC);

struct NoSFS : SFSScheme {
    void operator()(ParticleField& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};