#pragma once

#include "vpmcore/vpmmain.h"

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

int numberParticles(int Nphi, int nc, int extra_nc = 0);

int numberParticles(const VortexRing const &ring);

int addVortexRing(ParticleBuffer particleBuffer, vpmfloat circulation, vpmfloat R, vpmfloat Rcross,
    int Nphi, int nc, vpmfloat sigma, int extra_nc, vpmvec3 ringPosition,
    vpmmat3 ringOrientation, int startingIndex, int maxParticles);

int initVortexRings(ParticleBuffer particleBuffer, int maxParticles);