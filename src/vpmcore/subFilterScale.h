#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "particleField.h"

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcEstrNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel);
template <typename R, typename S, typename K>
__device__ void calcEstrNaive(int index, ParticleField<R, S, K>* field);
template <typename R, typename S, typename K>
__device__ void dynamicProcedure(int index, ParticleField<R, S, K>* field, float alpha, float relaxFactor,
                                 bool forcePositive, float minC, float maxC);

struct DynamicSFS {
    float minC;
    float maxC;
    float alpha;
    float relaxFactor;
    bool forcePositive;

    DynamicSFS(float minC = 0, float maxC = 1, float alpha = 0.667, float relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    template <typename R, typename S, typename K>
    __device__ void operator()(int index, ParticleField<R, S, K>* field, float a, float b);
};