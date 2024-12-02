#include "particlefield.h"
#include "particle.h"
#include "velocities.h"
#include "vpmUtils.h"
#include <glm/glm.hpp>
#include "relaxation.h"
#include "timeIntegration.h"

template <class Relax, class SFS>
__global__ void rungekutta(int N, ParticleField* field, float dt, bool relax, Relax relaxation, SFS sfs) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    glm::vec3 S;
    float Z;

    Particle& particle = field->particles[index];

    float zeta0 = field->kernel.zeta(0.0f);

    // Reset temp variable (necessary?)
    particle.M = glm::mat3{ 0 };

    float rungeKuttaCoefs[3][2] = {
        {0.0f, 1.0f / 3.0f},
        {-5.0f / 9.0f, 15.0f / 16.0f},
        {-153.0f / 128.0f, 8.0f / 15.0f}
    };

    // Loop over the pairs
    for (int i = 0; i < 3; ++i) {
        float a = rungeKuttaCoefs[i][0];
        float b = rungeKuttaCoefs[i][1];

        // RUN SFS
        sfs(field, a, b);

        __syncthreads();

        particle.M[0] = a * particle.M[0] + dt * (particle.U + field->Uinf);
        particle.X += b * particle.M[0];

        S = xDotNablaY(particle.Gamma, particle.J);
        Z = 0.2f * glm::dot(S, particle.Gamma) / glm::dot(particle.Gamma, particle.Gamma);

        particle.M[1] = a * particle.M[1] + dt * (S - 3 * Z * particle.Gamma - 
                        particle.C[0] * particle.SFS * p.sigma * p.sigma * p.sigma / zeta0);
        particle.M[2][1] = a * particle.M[2][1] - dt * (particle.sigma * Z);

        particle.Gamma += b * particle.M[1];
        particle.sigma += b * particle.M[2][1];

        __syncthreads();
    }

    if (relax) {
        // field.resetParticles();
        _reset_particles(field);

        calcVelJacNaive(index, field);

        relaxation(particle);
    }
}