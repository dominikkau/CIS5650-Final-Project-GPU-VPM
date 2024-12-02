#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "particle.h"
#include "subFilterScale.h"
#include "relaxation.h"
#include "kernels.h"

// ParticleField definition
template <typename R=PedrizzettiRelaxation, typename S=DynamicSFS, typename K=GaussianErfKernel>
class ParticleField {
public:
    // User inputs
    int maxParticles;                      // Maximum number of particles
    Particle* particles;                // Pointer to particle buffer
    // std::vector<void*> bodies;             // CHECK - Placeholder for ExaFMM array of bodies
    // F formulation;                         // VPM formulation
    // V viscous;                             // Viscous scheme

    // Internal properties
    int np;                            // Number of particles in the field
    int nt;                            // Current time step number
    float t;                          // Current time

    // Solver settings
    K kernel;                        // Vortex particle kernel
    // std::function<void()> UJ;              // Particle-to-particle calculation

    // Optional inputs
    glm::vec3 Uinf;             // Uniform freestream function Uinf(t)
    S SFS;                      // Subfilter-scale contributions scheme
    bool transposed;            // Transposed vortex stretch scheme
    R relaxation;             // Relaxation scheme
    // FMM fmm;                 // Fast-multipole settings


    // Constructor
    ParticleField(
        int maxparticles,
        // std::vector<void*> bodies,
        int np = 0,
        int nt = 0,
        float t = 0.0f,
        K kernel = GaussianErfKernel(),
        // std::function<void()> UJ=UJ_fmm,
        glm::vec3 Uinf = glm::vec3(0, 0, 0),
        S SFS = DynamicSFS(),
        bool transposed = true,
        R relaxation = PedrizzettiRelaxation(0.005))
        :
        //   particles(particles),
        //   bodies(bodies),
        //   formulation(formulation),
        //   viscous(viscous),
        maxParticles(maxparticles),
        np(np),
        nt(nt),
        t(t),
        kernel(kernel),
        Uinf(Uinf),
        SFS(SFS),
        transposed(transposed),
        relaxation(relaxation) {};
};

