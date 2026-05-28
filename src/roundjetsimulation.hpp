#pragma once

#include "vpmcore/vpmmain.h"

size_t addAnnulus(ParticleBuffer particleBuffer, vpm::real circulation, vpm::real R,
    int Nphi, vpm::real sigma, vpm::real area, vpm::vec3 jetOrigin,
    vpm::mat3 jetOrientation, bool isStatic, size_t startingIndex, size_t maxParticles);

std::pair<size_t, size_t> initRoundJet(ParticleBuffer particleBuffer,
    ParticleBuffer boundaryBuffer, size_t maxParticles);
