#pragma once

struct ParticleField;
struct Particle;

class PedrizzettiRelaxation {
public:
    float relaxFactor;

    __host__ __device__ PedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}

    __device__ void relax(int index, ParticleField* field);
};

class CorrectedPedrizzettiRelaxation {
public:
    float relaxFactor;

    __host__ __device__ CorrectedPedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}

    __device__ void relax(int index, ParticleField* field);
};