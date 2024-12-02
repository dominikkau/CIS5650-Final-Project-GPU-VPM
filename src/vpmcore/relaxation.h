#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct Particle;

struct PedrizzettiRelaxation {
    float relaxFactor;
    PedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}
    __device__ void operator()(Particle& particle);
};

struct CorrectedPedrizzettiRelaxation {
    float relaxFactor;
    CorrectedPedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}
    __device__ void operator()(Particle& particle);
};

struct NoRelaxation {
    __device__ void operator()(Particle& particle);
};