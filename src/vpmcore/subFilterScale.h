#pragma once

struct ParticleField;
class Kernel;

__device__ void calcEstrNaive(int index, ParticleField* source, ParticleField* target, Kernel& kernel);
__device__ void calcEstrNaive(int index, ParticleField* field);
__device__ void dynamicProcedure(int index, ParticleField* field, float alpha, float relaxFactor,
                                 bool forcePositive, float minC, float maxC);