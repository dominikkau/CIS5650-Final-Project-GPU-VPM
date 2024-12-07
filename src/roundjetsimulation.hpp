#pragma once

#include "vpmcore/kernel.h"


int addAnnulus(Particle* particleBuffer, vpmfloat circulation, vpmfloat R,
    int Nphi, vpmfloat sigma, vpmfloat area, vpmvec3 ringPosition,
    vpmmat3 ringOrientation, int startingIndex, int maxParticles);
