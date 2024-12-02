#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

const float PI = 3.14159265358979f;
const float const4 = 1 / (4 * PI);

struct ParticleField;
struct Kernel;

template <class Kernel>
__device__ void calcVelJacNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel);
__device__ void calcVelJacNaive(int index, ParticleField* field);