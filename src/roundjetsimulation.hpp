#pragma once

#include "vpmcore/vpmmain.h"

size_t addAnnulus(ParticleBuffer particleBuffer, vpmfloat circulation, vpmfloat R,
    int Nphi, vpmfloat sigma, vpmfloat area, vpmvec3 jetOrigin,
    vpmmat3 jetOrientation, bool isStatic, size_t startingIndex, size_t maxParticles);

std::pair<size_t, size_t> initRoundJet(ParticleBuffer particleBuffer,
    ParticleBuffer boundaryBuffer, size_t maxParticles);
