#include <glm/glm.hpp>
#include "particle.h"
#include "vpmUtils.h"
#include "relaxation.h"

__host__ __device__ void PedrizzettiRelaxation::operator()(Particle& particle) {
    glm::vec3 omega = nablaCrossX(particle.J);
    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * glm::length(particle.Gamma) / glm::length(omega) * omega;
}

__host__ __device__ void CorrectedPedrizzettiRelaxation::operator()(Particle& particle) {
    glm::vec3 omega = nablaCrossX(particle.J);
    float omegaNorm = glm::length(omega);
    float gammaNorm = glm::length(particle.Gamma);
    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * gammaNorm / omegaNorm * omega;
    particle.Gamma /= sqrt(1.0f - 2.0f * (1.0f - relaxFactor) * relaxFactor 
                      * (1.0f - glm::dot(particle.Gamma, omega) / (omegaNorm * gammaNorm)));
}

__host__ __device__ void NoRelaxation::operator()(Particle& particle) {}