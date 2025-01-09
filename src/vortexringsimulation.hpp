#pragma once

#include "vpmcore/kernel.h"

int numberParticles(int Nphi, int nc, int extra_nc = 0);

int addVortexRing(Particle* particleBuffer, vpmfloat circulation, vpmfloat R, vpmfloat Rcross,
    int Nphi, int nc, vpmfloat sigma, int extra_nc, vpmvec3 ringPosition,
    vpmmat3 ringOrientation, int startingIndex, int maxParticles);

int initVortexRings(Particle* particleBuffer, int maxParticles);
int initVortexRingsCoal(CoalescedParticle particleBuffer, int maxParticles);