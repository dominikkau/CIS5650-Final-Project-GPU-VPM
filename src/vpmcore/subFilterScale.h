#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct ParticleField;
//struct Kernel;

template <class Kernel>
__device__ void calcEstrNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel);
__device__ void calcEstrNaive(int index, ParticleField* field);
__device__ void dynamicProcedure(int index, ParticleField* field, float alpha, float relaxFactor,
                                 bool forcePositive, float minC, float maxC);

struct DynamicSFS {
    float minC;
    float maxC;
    float alpha;
    float relaxFactor;
    bool forcePositive;

    DynamicSFS(float minC = 0, float maxC = 1, float alpha = 0.667, float relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    __device__ void operator()(int index, ParticleField* field, float a = 1.0f, float b = 1.0f);
};