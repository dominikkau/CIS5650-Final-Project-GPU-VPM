#include <glm/glm.hpp>
#include "particlefield.h"
#include "particle.h"
#include "vpmUtils.h"
#include "velocities.h"
#include "subFilterScale.h"

template <typename Kernel>
__device__ void calcEstrNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel) {
    Particle& targetParticle = target->particles[index];

    for (int i = 0; i < source->np; ++i) {
        Particle& sourceParticle = source->particles[i];

        glm::vec3 S = xDotNablaY(sourceParticle.Gamma, targetParticle.J - sourceTarget.J);

        glm::vec3 dX = targetParticle.X - sourceParticle.X;
        float r = glm::length(dX);

        targetParticle.SFS += kernel.zeta(r / sourceParticle.sigma) / powf(sourceParticle.sigma, 3.0f) * S;
    }
}

__device__ void calcEstrNaive(int index, ParticleField* field) {
    calcEstrNaive(index, field, field, field->kernel);
}

__device__ void dynamicProcedure(int index, ParticleField* field, float alpha, float relaxFactor,
                                 bool forcePositive, float minC, float maxC) {
    Particle& particle = field->particles[index];

    // CALCULATIONS WITH TEST FILTER
    particle.sigma *= alpha;

    // field.resetParticles();
    _reset_particles(field);
    calcVelJacNaive(index, field);

    // field.resetParticlesSFS();
    _reset_particles_sfs(field);
    calcEstrNaive(index, field);

    // Clear temporary variable (really necessary?)
    particle.M = 0;

    // temporary variables
    particle.M[0] = xDotNablaY(particle.Gamma, particle.J);
    particle.M[1] = particle.SFS;

    // CALCULATIONS WITH DOMAIN FILTER
    particle.sigma /= alpha;
    // field.resetParticles();
    _reset_particles(field);
    calcVelJacNaive(index, field);

    // field.resetParticlesSFS();
    _reset_particles_sfs(field);
    calcEstrNaive(index, field);

    // Save temporary variables
    particle.M[0] -= xDotNablaY(particle.Gamma, particle.J);
    particle.M[1] -= particle.SFS;

    // CALCULATE COEFFICIENT
    float numerator = glm::dot(particle.M[0], particle.Gamma);
    numerator *= 3.0f * alpha - 2.0f;

    float denominator = glm::dot(particle.M[1], particle.Gamma);
    denominator *= particle.sigma * particle.sigma * particle.sigma / field->kernel.zeta(0);

    // Don't initialize denominator to 0
    if (particle.C[2] == 0) particle.C[2] = denominator;

    // Lagrangian average
    numerator   = relaxFactor * numerator   + (1 - relaxFactor) * particle.C[1];
    denominator = relaxFactor * denominator + (1 - relaxFactor) * particle.C[2];

    // Enforce maximum and minimum absolute values
    if (fabs(numerator/denominator) > maxC) {
        if (fabs(denominator) < fabs(particle.C[2])) denominator = copysign(particle.C[2], denominator);

        if (fabs(numerator/denominator) > maxC) numerator = copysign(denominator, numerator) * maxC;
    }
    else if (fabs(numerator/denominator) < minC) numerator = copysign(denominator, numerator) * minC;

    // Save numerator and denominator of model coefficient
    particle.C[1] = numerator;
    particle.C[2] = denominator;

    // Store model coefficient
    particle.C[0] = particle.C[1] / particle.C[2];

    // Force the coefficient to be positive
    if (forcePositive) particle.C[0] = fabs(particle.C[0]);

    // Clear temporary variable
    particle.M = 0;
}

__host__ __device__ void DynamicSFS::operator()(int index, ParticleField* field, float a=1.0f, float b=1.0f) {
    Particle& particle = field->particles[index];

    if (a == 1.0f || a == 0.0f) {
        dynamicProcedure(index, field, alpha, relaxFactor, forcePositive, minC, maxC);

        if (particle.C[0] * glm::dot(particle.Gamma, particle.SFS) < 0) particle.C[0] = 0;
    }
    else {
        _reset_particles(field);
        calcVelJacNaive(index, field);

        _reset_particles_sfs(field);
        calcEstrNaive(index, field);
    }
}