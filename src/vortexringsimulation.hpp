#pragma once

#include "vpmcore/vpmmain.h"

namespace vortex_rings {
    struct VortexRing {
        vpmfloat circulation = 1.0f;
        vpmfloat R = 1.0f;
        vpmfloat Rcross = 0.1f;
        vpmfloat sigma = 0.1f;
        int Nphi = 100;
        int nc = 3;
        int extra_nc = 0;
        vpmvec3 position{ 0, 0, 0 };
        vpmmat3 orientation{ 1.0f };
    };

    size_t numberParticles(int Nphi, int nc, int extra_nc = 0);

    size_t numberParticles(const VortexRing &ring);

    size_t numberParticles(const std::vector<VortexRing>& rings);

    size_t addVortexRing(ParticleBuffer particleBuffer, vpmfloat circulation, vpmfloat R, vpmfloat Rcross,
        int Nphi, int nc, vpmfloat sigma, int extra_nc, vpmvec3 ringPosition,
        vpmmat3 ringOrientation, size_t startingIndex);

    size_t initVortexRings(ParticleBuffer particleBuffer);

    size_t initParticleBuffer(ParticleBuffer particleBuffer, const std::vector<VortexRing> &rings);
}