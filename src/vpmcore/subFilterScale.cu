#include <glm/glm.hpp>
#include "particlefield.h"
#include "particle.h"
#include "vpmUtils.h"

template <typename Kernel>
__device__ void calcEstrNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel)
{
    Particle& targetParticle = target.particles[index];

    for (int i = 0; i < source.np; ++i)
    {
        Particle& sourceParticle = source.particles[i];

        glm::vec3 S = xDotNablaY(sourceParticle.Gamma, targetParticle.J - sourceTarget.J);

        glm::vec3 dX = targetParticle.X - sourceParticle.X;
        float r = glm::length(dX);

        targetParticle.SFS = kernel.zeta(r / sourceParticle.sigma) / powf(sourceParticle.sigma, 3.0f) * S;
    }
}