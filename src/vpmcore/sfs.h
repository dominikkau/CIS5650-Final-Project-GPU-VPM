#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

class ParticleField;
class ParticleBuffer;

struct SFSScheme {
    virtual void operator()(ParticleField& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize, cudaStream_t stream = 0) = 0;
};

class DynamicSFS : public SFSScheme {
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

class NoSFS : public SFSScheme {
    void operator()(ParticleField& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize, cudaStream_t stream = 0);
};