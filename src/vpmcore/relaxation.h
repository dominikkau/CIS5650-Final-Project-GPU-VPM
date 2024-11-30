#pragma once

__device__ void relax_pedrizzetti(int index, ParticleField* field, float relaxFactor);
__device__ void relax_correctedpedrizzetti(int index, ParticleField* field, float relaxFactor);