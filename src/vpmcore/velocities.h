#pragma once

constexpr float const4 = 1 / (4 * math.pi);

__device__ void calcVelJacNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel);