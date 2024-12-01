#pragma once

constexpr float PI = 3.14159265358979f;
constexpr float const4 = 1 / (4 * PI);

struct ParticleField;
struct Kernel;

template <class Kernel>
__device__ void calcVelJacNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel);
__device__ void calcVelJacNaive(int index, ParticleField* field);