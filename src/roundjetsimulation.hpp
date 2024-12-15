#pragma once

#include "vpmcore/kernel.h"

int addAnnulus(Particle* particleBuffer, vpmfloat circulation, vpmfloat R,
    int Nphi, vpmfloat sigma, vpmfloat area, vpmvec3 jetOrigin,
    vpmmat3 jetOrientation, bool isStatic, int startingIndex, int maxParticles);

std::pair<int, int> initRoundJet(Particle* particleBuffer, Particle* boundaryBuffer, int maxParticles);
