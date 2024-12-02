#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct ParticleField;
struct Kernel;

template <class Kernel>
__device__ void calcVelJacNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel);
__device__ void calcVelJacNaive(int index, ParticleField* field);