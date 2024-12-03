#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include "kernel.h"
#include "../lean_vtk.hpp"
#include "../vortexringsimulation.hpp"

// Constructor
Particle::Particle() 
    : X(0.0f), Gamma(0.0f), sigma(0.0f), vol(0.0f), circulation(0.0f), isStatic(false),
      U(0.0f), J(0.0f), PSE(0.0f), M(0.0f), C(0.0f), SFS(0.0f) {}

__host__ __device__ void Particle::reset() {
    U = glm::vec3(0.0f);
    J = glm::mat3(0.0f);
    PSE = glm::vec3(0.0f);
}

__host__ __device__ void Particle::resetSFS() {
    SFS = glm::vec3(0.0f);
}

__device__ void PedrizzettiRelaxation::operator()(Particle& particle) {
    glm::vec3 omega = nablaCrossX(particle.J);
    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * glm::length(particle.Gamma) / glm::length(omega) * omega;
}

 __device__ void CorrectedPedrizzettiRelaxation::operator()(Particle& particle) {
    glm::vec3 omega = nablaCrossX(particle.J);
    float omegaNorm = glm::length(omega);
    float gammaNorm = glm::length(particle.Gamma);
    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * gammaNorm / omegaNorm * omega;
    particle.Gamma /= sqrt(1.0f - 2.0f * (1.0f - relaxFactor) * relaxFactor 
                      * (1.0f - glm::dot(particle.Gamma, omega) / (omegaNorm * gammaNorm)));
}

__device__ void NoRelaxation::operator()(Particle& particle) {}

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcEstrNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel) {
    Particle& targetParticle = target->particles[index];

    for (int i = 0; i < source->np; ++i) {
        Particle& sourceParticle = source->particles[i];

        glm::vec3 S = xDotNablaY(sourceParticle.Gamma, targetParticle.J - sourceParticle.J);

        glm::vec3 dX = targetParticle.X - sourceParticle.X;
        float r = glm::length(dX);

        targetParticle.SFS += kernel.zeta(r / sourceParticle.sigma) / powf(sourceParticle.sigma, 3.0f) * S;
    }
}

template <typename R, typename S, typename K>
__device__ void calcEstrNaive(int index, ParticleField<R, S, K>* field) {
    calcEstrNaive(index, field, field, field->kernel);
}

template <typename R, typename S, typename K>
__device__ void dynamicProcedure(int index, ParticleField<R, S, K>* field, float alpha, float relaxFactor,
                                 bool forcePositive, float minC, float maxC) {
    Particle& particle = field->particles[index];

    // CALCULATIONS WITH TEST FILTER
    particle.sigma *= alpha;

    particle.reset();
    calcVelJacNaive(index, field);

    particle.resetSFS();
    calcEstrNaive(index, field);

    // Clear temporary variable (really necessary?)
    particle.M = glm::mat3{ 0.0f };

    // temporary variables
    particle.M[0] = xDotNablaY(particle.Gamma, particle.J);
    particle.M[1] = particle.SFS;

    // CALCULATIONS WITH DOMAIN FILTER
    particle.sigma /= alpha;

    particle.reset();
    calcVelJacNaive(index, field);

    particle.resetSFS();
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
    particle.M = glm::mat3{ 0.0f };
}

template <typename R, typename S, typename K>
__device__ void DynamicSFS::operator()(int index, ParticleField<R, S, K>* field, float a, float b) {
    Particle& particle = field->particles[index];

    if (a == 1.0f || a == 0.0f) {
        dynamicProcedure(index, field, alpha, relaxFactor, forcePositive, minC, maxC);

        if (particle.C[0] * glm::dot(particle.Gamma, particle.SFS) < 0) particle.C[0] = 0;
    }
    else {
        particle.reset();
        calcVelJacNaive(index, field);

        particle.resetSFS();
        calcEstrNaive(index, field);
    }
}


template <typename R, typename S, typename K>
__device__ void NoSFS::operator()(int index, ParticleField<R, S, K>* field, float a, float b) {
    Particle& particle = field->particles[index];

    particle.reset();
    calcVelJacNaive(index, field);
}

template <typename R, typename S, typename K>
__global__ void rungekutta(int N, ParticleField<R, S, K>* field, float dt, bool relax) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    glm::vec3 S;
    float Z;

    Particle& particle = field->particles[index];

    float zeta0 = field->kernel.zeta(0.0f);

    // Reset temp variable (necessary?)
    particle.M = glm::mat3{ 0.0f };

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
        field->SFS(index, field, a, b);

        __syncthreads();

        particle.M[0] = a * particle.M[0] + dt * (particle.U + field->Uinf);
        particle.X += b * particle.M[0];

        S = xDotNablaY(particle.Gamma, particle.J);
        Z = 0.2f * glm::dot(S, particle.Gamma) / glm::dot(particle.Gamma, particle.Gamma);

        particle.M[1] = a * particle.M[1] + dt * (S - 3 * Z * particle.Gamma - 
                        particle.C[0] * particle.SFS * particle.sigma * particle.sigma * particle.sigma / zeta0);
        particle.M[2][1] = a * particle.M[2][1] - dt * (particle.sigma * Z);

        particle.Gamma += b * particle.M[1];
        particle.sigma += b * particle.M[2][1];

        __syncthreads();
    }

    if (relax) {
        particle.reset();
        calcVelJacNaive(index, field);

        field->relaxation(particle);
    }
}

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel) {
    Particle& targetParticle = target->particles[index];

    for (int i = 0; i < source->np; ++i) {
        if (i == index) continue;

        Particle& sourceParticle = source->particles[i];

        glm::vec3 dX = targetParticle.X - sourceParticle.X;
        float r = glm::length(dX);
        float r3 = r*r*r;

        // is this needed?
        if (r < EPS) continue;

        // Kernel evaluation
        float g_sgm = kernel.g(r / sourceParticle.sigma);
        float dg_sgmdr = kernel.dgdr(r / sourceParticle.sigma);

        // Compute velocity
        glm::vec3 crossProd = glm::cross(dX, sourceParticle.Gamma) * (-const4 / r3);
        targetParticle.U += g_sgm * crossProd;

        // Compute Jacobian
        float tmp = dg_sgmdr / (sourceParticle.sigma * r) - 3.0f * g_sgm / (r*r);
        glm::vec3 dX_norm = dX / r;

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                targetParticle.J[l][k] += tmp * crossProd[k] * dX_norm[l];
            }
        }

        tmp = - const4 * g_sgm / r3;

        // Account for kronecker delta term
        targetParticle.J[0][1] -= tmp * sourceParticle.Gamma[2];
        targetParticle.J[0][2] += tmp * sourceParticle.Gamma[1];
        targetParticle.J[1][0] += tmp * sourceParticle.Gamma[2];
        targetParticle.J[1][2] -= tmp * sourceParticle.Gamma[0];
        targetParticle.J[2][0] -= tmp * sourceParticle.Gamma[1];
        targetParticle.J[2][1] += tmp * sourceParticle.Gamma[0];
    }
}

template <typename R, typename S, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<R, S, K>* field) {
    calcVelJacNaive(index, field, field, field->kernel);
}

void writeVTK(int numParticles, Particle* particleBuffer, std::string filename, int timestep) {
    const int dim = 3;

    leanvtk::VTUWriter writer;

    std::vector<double> particleX;
    std::vector<double> particleU;
    std::vector<double> particleSigma;
    std::vector<double> particleIdx;
    particleX.reserve(dim * numParticles);
    particleU.reserve(dim * numParticles);
    particleSigma.reserve(numParticles);
    particleIdx.reserve(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        Particle& particle = particleBuffer[i];

        particleIdx.push_back(i);
        particleSigma.push_back(particle.sigma);

        for (int j = 0; j < dim; ++j) {
            particleU.push_back(particle.U[j]);
            particleX.push_back(particle.X[j]);
        }
    }

    writer.add_scalar_field("index", particleIdx);
    writer.add_scalar_field("sigma", particleSigma);
    writer.add_vector_field("position", particleX, dim);
    writer.add_vector_field("velocity", particleU, dim);
    writer.write_point_cloud("../output/" + filename + "_" + std::to_string(timestep) + ".vtu", dim, particleX);
}

template <typename R, typename S, typename K>
void runVPM(
    int maxParticles,
    int numParticles,
    int numTimeSteps,
    float dt,
    int fileSaveSteps,
    int blockSize,
    glm::vec3 uInf,
    Particle* particleBuffer,
    R relaxation,
    S sfs,
    K kernel) {
    int fullBlocksPerGrid{ (numParticles + blockSize - 1) / blockSize };

    // Declare device particle buffer
    Particle* dev_particleBuffer;
    cudaMalloc((void**)&dev_particleBuffer, maxParticles * sizeof(Particle));

    // Copy particle buffer from host to device
    cudaMemcpy(dev_particleBuffer, particleBuffer, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    ParticleField<R, S, K> field{
        maxParticles,
        dev_particleBuffer,
        numParticles,
        0,
        0.0f,
        kernel,
        uInf,
        sfs,
        true,
        relaxation
    };

    // Declare device particle field and copy host field to device
    ParticleField<R, S, K>* dev_field;
    cudaMalloc((void**)&dev_field, sizeof(ParticleField<R, S, K>));
    cudaMemcpy(dev_field, &field, sizeof(ParticleField<R, S, K>), cudaMemcpyHostToDevice);

    std::cout << particleBuffer[0].U.x << std::endl;

    for (int i = 0; i < numTimeSteps; ++i) {
        rungekutta<R, S, K><<<fullBlocksPerGrid, blockSize>>>(
            numParticles, dev_field, dt, true
        );

        //cudaMemcpy(&field, dev_field, sizeof(ParticleField<R, S, K>), cudaMemcpyDeviceToHost);
        cudaMemcpy(particleBuffer, dev_particleBuffer, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

        if (i % fileSaveSteps == 0) {
            writeVTK(numParticles, particleBuffer, "test", i / fileSaveSteps);
            std::cout << particleBuffer[0].U.x << std::endl;
        }
    }

    // free device memory
    cudaFree(dev_particleBuffer);
    cudaFree(dev_field);
}

void randomCubeInit(Particle* particleBuffer, int N, float cubeSize, float maxCirculation, float maxSigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::uniform_real_distribution<float> uniformPos(0, 1);

    for (int i = 0; i < N; ++i) {
        Particle& particle = particleBuffer[i];

        particle.sigma = maxSigma * uniformPos(gen);
        particle.Gamma = maxCirculation * uniform(gen) * glm::normalize(glm::vec3{ uniform(gen), uniform(gen), uniform(gen) });
        particle.circulation = glm::length(particle.Gamma);

        particle.X = cubeSize * uniform(gen) * glm::normalize(glm::vec3{ uniform(gen), uniform(gen), uniform(gen) });
    }
}

void randomSphereInit(Particle* particleBuffer, int N, float sphereRadius, float maxCirculation, float maxSigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::uniform_real_distribution<float> uniformPos(0, 1);

    for (int i = 0; i < N; ++i) {
        Particle& particle = particleBuffer[i];

        particle.sigma = maxSigma * uniformPos(gen);
        particle.Gamma = maxCirculation * uniform(gen) * glm::normalize(glm::vec3{ uniform(gen), uniform(gen), uniform(gen) });
        particle.circulation = glm::length(particle.Gamma);

        float theta = 2 * PI * uniformPos(gen);
        float phi = PI * uniformPos(gen);
        float radius = std::cbrt(uniformPos(gen)) * sphereRadius;

        float x = radius * sin(phi) * cos(theta);
        float y = radius * sin(phi) * sin(theta);
        float z = radius * cos(phi);
        particle.X = glm::vec3{ x, y, z };
    }
}

void runSimulation() {
    // Define basic parameters
    int maxParticles{ 1000 };
    int numTimeSteps{ 10 };
    float dt{ 0.01f };
    int numBlocks{ 128 };
    int numStepsVTK{ 10 };
    glm::vec3 uInf{ 0, 0, 0 };

    // Create host particle buffer
    Particle* particleBuffer = new Particle[maxParticles];
    // Initialize particle buffer
    //randomSphereInit(particleBuffer, maxParticles, 10.0f, 1.0f, 0.5f);
    initVortexRings(particleBuffer, maxParticles);

    // Run VPM method
    runVPM(
        maxParticles,
        maxParticles,
        numTimeSteps,
        dt,
        numStepsVTK,
        numBlocks,
        uInf,
        particleBuffer,
        PedrizzettiRelaxation(0.005f),
        DynamicSFS(),
        GaussianErfKernel()
    );

    // Free host particle buffer
    delete[] particleBuffer;
}