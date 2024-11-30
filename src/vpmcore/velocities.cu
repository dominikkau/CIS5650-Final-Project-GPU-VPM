#include <glm/glm.hpp>
#include "particlefield.h"
#include "particle.h"
#include "kernels.h"

#define EPS 1e-6f

__device__ void calcVelJacNaive(int index, ParticleField* source, ParticleField* target, Kernel kernel)
{
    Particle& targetParticle = target.particles[index];

    for (int i = 0; i < source.np; ++i)
    {
        if (i == index) continue;

        Particle& sourceParticle = source.particles[i];

        glm::vec3 dX = targetParticle.X - sourceParticle.X;
        float r = glm::length(dX);

        // is this needed?
        if (r < EPS) continue;

        // Kernel evaluation
        float g_sgm = kernel.g(r / sourceParticle.sigma);
        float dg_sgmdr = kernel.dgdr(r / sourceParticle.sigma);

        // Compute velocity
        glm::vec3 crossProd = (-const4 / powf(r, 3.0f)) * glm::cross(dX, sourceParticle.Gamma);
        targetParticle.U += g_sgm * crossProd;

        // Compute Jacobian
        float aux = dg_sgmdr / (sourceParticle.sigma * r) - 3.0f * g_sgm / powf(r, 2.0f);
        glm::vec3 dX_norm = dX / r;

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                targetParticle.J[k][l] += aux * crossProd[k] * dX_norm[l];
            }
        }

        aux = -const4 * g_sgm / powf(r, 3.0f);

        // Account for kronecker delta term
        targetParticle.J[1][0] -= aux * sourceParticle.Gamma[2];
        targetParticle.J[2][0] += aux * sourceParticle.Gamma[1];
        targetParticle.J[0][1] += aux * sourceParticle.Gamma[2];
        targetParticle.J[2][1] -= aux * sourceParticle.Gamma[0];
        targetParticle.J[0][2] -= aux * sourceParticle.Gamma[1];
        targetParticle.J[1][2] += aux * sourceParticle.Gamma[0];
    }
}