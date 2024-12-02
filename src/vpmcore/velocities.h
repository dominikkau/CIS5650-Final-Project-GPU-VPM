#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "particleField.h"

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel);

template <typename R, typename S, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<R, S, K>* field);