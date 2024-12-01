#pragma once

struct ParticleField;

template <class Relax, class SFS>
__global__ void rungekutta(int N, ParticleField* field, float dt, bool relax, Relax relaxation, SFS sfs)