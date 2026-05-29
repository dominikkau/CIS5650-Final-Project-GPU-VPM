#pragma once

#include "vpmcore/vpmmain.h"
#include "vpmcore/common.h"

namespace vortex_rings {
    struct VortexRing {
        vpm::real circulation = 1.0f;
        vpm::real R = 1.0f;
        vpm::real Rcross = 0.1f;
        vpm::real sigma = 0.1f;
        int Nphi = 100;
        int nc = 3;
        int extra_nc = 0;
        vpm::vec3 position{ 0, 0, 0 };
        vpm::mat3 orientation{ 1.0f };
    };

    vpm::pidx_t numberParticles(int Nphi, int nc, int extra_nc = 0);

    vpm::pidx_t numberParticles(const VortexRing& ring);

    vpm::pidx_t numberParticles(const std::vector<VortexRing>& rings);

    vpm::pidx_t addVortexRing(ParticleBuffer& particleBuffer, vpm::real circulation, vpm::real R, vpm::real Rcross,
        int Nphi, int nc, vpm::real sigma, int extra_nc, vpm::vec3 ringPosition,
        vpm::mat3 ringOrientation, vpm::pidx_t startingIndex);

    vpm::pidx_t initVortexRings(ParticleBuffer& particleBuffer);

    vpm::pidx_t initParticleBuffer(ParticleBuffer& particleBuffer, const std::vector<VortexRing>& rings);
}