#pragma once

#include "vpmcore/vpmmain.h"
#include "vpmcore/common.h"

vpm::pidx_t addAnnulus(ParticleBuffer particleBuffer, vpm::real circulation, vpm::real R,
    int Nphi, vpm::real sigma, vpm::real area, vpm::vec3 jetOrigin,
    vpm::mat3 jetOrientation, bool isStatic, vpm::pidx_t startingIndex, vpm::pidx_t maxParticles);

std::pair<vpm::pidx_t, vpm::pidx_t> initRoundJet(ParticleBuffer particleBuffer,
    ParticleBuffer boundaryBuffer, vpm::pidx_t maxParticles);
