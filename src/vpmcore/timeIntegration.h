#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "particleField.h"

template <typename R, typename S, typename K>
__global__ void rungekutta(int N, ParticleField<R, S, K>* field, float dt, bool relax);

void runVPM();