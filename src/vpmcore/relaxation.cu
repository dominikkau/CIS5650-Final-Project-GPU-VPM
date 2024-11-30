#include <glm/glm.hpp>
#include "particlefield.h"
#include "particle.h"
#include "vpmUtils.h"

__device__ void relax_pedrizzetti(int index, ParticleField* field, float relaxFactor)
{
    Particle& particle = field.particles[index];

    glm::vec3 omega = nablaCrossX(particle.J);

    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * glm::length(particle.Gamma) / glm::length(omega) * omega;
}

__device__ void relax_correctedpedrizzetti(int index, ParticleField* field, float relaxFactor)
{
    Particle& particle = field.particles[index];

    glm::vec3 omega = nablaCrossX(particle.J);

    float omegaNorm = glm::length(omega);
    float gammaNorm = glm::length(particle.Gamma);

    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * gammaNorm / omegaNorm * omega;

    particleGamma /= sqrtf(1.0f - 2.0f * (1.0f - relaxFactor) * relaxFactor 
                           * (1.0f - glm::dot(particle.Gamma, omega) / (omegaNorm * gammaNorm)));
}