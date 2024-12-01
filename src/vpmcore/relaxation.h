#pragma once

struct Particle;

struct PedrizzettiRelaxation {
    float relaxFactor;
    PedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}
    __host__ __device__ void operator()(int index, Particle& particle);
};

struct CorrectedPedrizzettiRelaxation {
    float relaxFactor;
    CorrectedPedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}
    __host__ __device__ void operator()(int index, Particle& particle);
};

struct NoRelaxation {
    __host__ __device__ void operator()(int index, Particle& particle);
};