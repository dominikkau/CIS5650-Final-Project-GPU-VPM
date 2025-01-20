#pragma once

#include "vpmcore/kernel.h"

unsigned int addAnnulus(ParticleBuffer particleBuffer, vpmfloat circulation, vpmfloat R,
    int Nphi, vpmfloat sigma, vpmfloat area, vpmvec3 jetOrigin,
    vpmmat3 jetOrientation, bool isStatic, unsigned int startingIndex, unsigned int maxParticles);

std::pair<unsigned int, unsigned int> initRoundJet(ParticleBuffer particleBuffer, ParticleBuffer boundaryBuffer, unsigned int maxParticles);
