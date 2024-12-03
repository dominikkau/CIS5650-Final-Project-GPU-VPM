#pragma once

#include "vpmcore/kernel.h"

int numberParticles(int Nphi, int nc, int extra_nc = 0);

int addVortexRing(Particle* particleBuffer, float circulation, float R, float Rcross,
    int Nphi, int nc, float sigmas, int extra_nc, glm::vec3 ringPosition,
    glm::mat3 ringOrientation, int startingIndex, int maxParticles);

int initVortexRings(Particle* particleBuffer, int maxParticles);