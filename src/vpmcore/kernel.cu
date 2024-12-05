#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include "kernel.h"
#include "../lean_vtk.hpp"
#include "../vortexringsimulation.hpp"

__constant__ vpmfloat rungeKuttaCoefs[3][2] = {
    {0.0f, 1.0f / 3.0f},
    {-5.0f / 9.0f, 15.0f / 16.0f},
    {-153.0f / 128.0f, 8.0f / 15.0f}
};

#ifdef SHARED_MEMORY
extern __shared__ Particle s_particleBuffer[];
#endif

// Constructor
Particle::Particle() 
    : X(0.0f), Gamma(0.0f), sigma(0.0f), vol(0.0f), circulation(0.0f), isStatic(false),
      U(0.0f), J(0.0f), PSE(0.0f), M(0.0f), C(0.0f), SFS(0.0f) {}

__host__ __device__ void Particle::reset() {
    U = vpmvec3(0.0f);
    J = vpmmat3(0.0f);
    PSE = vpmvec3(0.0f);
}

__host__ __device__ void Particle::resetSFS() {
    SFS = vpmvec3(0.0f);
}

__device__ void PedrizzettiRelaxation::operator()(Particle& particle) {
    vpmvec3 omega = nablaCrossX(particle.J);
    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * glm::length(particle.Gamma) / glm::length(omega) * omega;
}

 __device__ void CorrectedPedrizzettiRelaxation::operator()(Particle& particle) {
    const vpmvec3 omega = nablaCrossX(particle.J);
    const vpmfloat omegaNorm = glm::length(omega);
    const vpmfloat gammaNorm = glm::length(particle.Gamma);
    particle.Gamma = (1.0f - relaxFactor) * particle.Gamma
                     + relaxFactor * gammaNorm / omegaNorm * omega;
    particle.Gamma /= sqrt(1.0f - 2.0f * (1.0f - relaxFactor) * relaxFactor 
                      * (1.0f - glm::dot(particle.Gamma, omega) / (omegaNorm * gammaNorm)));
}

__device__ void NoRelaxation::operator()(Particle& particle) {}

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcEstrNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel) {
#ifdef SHARED_MEMORY
    Particle& targetParticle = s_particleBuffer[threadIdx.x];
#else
    Particle& targetParticle = target->particles[index];
#endif

    for (int i = 0; i < source->np; ++i) {
        Particle& sourceParticle = source->particles[i];
        const vpmfloat sourceSigma = sourceParticle.sigma;

        targetParticle.SFS += kernel.zeta(glm::length(targetParticle.X - sourceParticle.X) / sourceSigma)
            / (sourceSigma * sourceSigma * sourceSigma)
            * xDotNablaY(sourceParticle.Gamma, targetParticle.J - sourceParticle.J);
    }
}

template <typename R, typename S, typename K>
__device__ void calcEstrNaive(int index, ParticleField<R, S, K>* field) {
    calcEstrNaive(index, field, field, field->kernel);
}

template <typename R, typename S, typename K>
__device__ void dynamicProcedure(int index, ParticleField<R, S, K>* field, vpmfloat alpha, vpmfloat relaxFactor,
                                 bool forcePositive, vpmfloat minC, vpmfloat maxC) {
#ifdef SHARED_MEMORY
    Particle& particle = s_particleBuffer[threadIdx.x];
#else
    Particle& particle = field->particles[index];
#endif

    const vpmfloat zeta0 = field->kernel.zeta(0);

    // CALCULATIONS WITH TEST FILTER
    field->particles[index].sigma *= alpha;
    __syncthreads();

    particle.reset();
    calcVelJacNaive(index, field);
    __syncthreads();

    particle.resetSFS();
    calcEstrNaive(index, field);

    // Clear temporary variable (not necessary?)
    // particle.M = vpmmat3{ 0.0f };

    // temporary variables
    particle.M[0] = xDotNablaY(particle.Gamma, particle.J);
    particle.M[1] = particle.SFS;

    // CALCULATIONS WITH DOMAIN FILTER
    field->particles[index].sigma /= alpha;
    __syncthreads();

    particle.reset();
    calcVelJacNaive(index, field);
    __syncthreads();

    particle.resetSFS();
    calcEstrNaive(index, field);

    // Save temporary variables
    particle.M[0] -= xDotNablaY(particle.Gamma, particle.J);
    particle.M[1] -= particle.SFS;

    // CALCULATE COEFFICIENT
    vpmfloat numerator = glm::dot(particle.M[0], particle.Gamma);
    numerator *= 3.0f * alpha - 2.0f;

    vpmfloat denominator = glm::dot(particle.M[1], particle.Gamma);
    denominator *= particle.sigma * particle.sigma * particle.sigma / zeta0;

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

    // Clear temporary variable (not necessary?)
    // particle.M = vpmmat3{ 0.0f };
}

template <typename R, typename S, typename K>
__device__ void DynamicSFS::operator()(int index, ParticleField<R, S, K>* field, vpmfloat a, vpmfloat b) {
#ifdef SHARED_MEMORY
    Particle& particle = s_particleBuffer[threadIdx.x];
#else
    Particle& particle = field->particles[index];
#endif

    if (a == 1.0f || a == 0.0f) {
        dynamicProcedure(index, field, alpha, relaxFactor, forcePositive, minC, maxC);

        if (particle.C[0] * glm::dot(particle.Gamma, particle.SFS) < 0) particle.C[0] = 0;
    }
    else {
        particle.reset();
        calcVelJacNaive(index, field);
        __syncthreads();

        particle.resetSFS();
        calcEstrNaive(index, field);
    }
}

template <typename R, typename S, typename K>
__device__ void NoSFS::operator()(int index, ParticleField<R, S, K>* field, vpmfloat a, vpmfloat b) {
#ifdef SHARED_MEMORY
    Particle& particle = s_particleBuffer[threadIdx.x];
#else
    Particle& particle = field->particles[index];
#endif

    particle.resetSFS();
    particle.reset();
    calcVelJacNaive(index, field);
}

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel) {
#ifdef SHARED_MEMORY
    Particle& targetParticle = s_particleBuffer[threadIdx.x];
#else
    Particle& targetParticle = target->particles[index];
#endif

    for (int i = 0; i < source->np; ++i) {
        if (i == index) continue;

        Particle& sourceParticle = source->particles[i];
        vpmfloat invSourceSigma = 1 / sourceParticle.sigma;
        vpmvec3 sourceGamma = sourceParticle.Gamma;

        vpmvec3 dX = targetParticle.X - sourceParticle.X;
        vpmfloat r = glm::length(dX);
        vpmfloat r3 = r*r*r;

        // is this needed?
        if (r < EPS) continue;

        // Kernel evaluation
        vpmfloat g_sgm = kernel.g(r * invSourceSigma);
        vpmfloat dg_sgmdr = kernel.dgdr(r * invSourceSigma);

        // Compute velocity
        vpmvec3 crossProd = glm::cross(dX, sourceGamma) * (-const4 / r3);
        targetParticle.U += g_sgm * crossProd;

        // Compute Jacobian
        vpmfloat tmp = dg_sgmdr * invSourceSigma / r - 3.0f * g_sgm / (r*r);
        vpmvec3 dX_norm = dX / r;

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                targetParticle.J[l][k] += tmp * crossProd[k] * dX_norm[l];
            }
        }

        tmp = - const4 * g_sgm / r3;

        // Account for kronecker delta term
        targetParticle.J[0][1] -= tmp * sourceGamma[2];
        targetParticle.J[0][2] += tmp * sourceGamma[1];
        targetParticle.J[1][0] += tmp * sourceGamma[2];
        targetParticle.J[1][2] -= tmp * sourceGamma[0];
        targetParticle.J[2][0] -= tmp * sourceGamma[1];
        targetParticle.J[2][1] += tmp * sourceGamma[0];
    }

#ifdef SHARED_MEMORY
    // Copy variables back to global memory
    target->particles[index].U = targetParticle.U;
    target->particles[index].J = targetParticle.J;
#endif
}

template <typename R, typename S, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<R, S, K>* field) {
    calcVelJacNaive(index, field, field, field->kernel);
}

template <typename R, typename S, typename K>
__global__ void rungekutta(int N, ParticleField<R, S, K>* field, vpmfloat dt, bool useRelax) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

#ifdef SHARED_MEMORY
    s_particleBuffer[threadIdx.x] = field->particles[index];
    Particle& particle = s_particleBuffer[threadIdx.x];
#else
    Particle& particle = field->particles[index];
#endif

    vpmvec3 S;
    vpmfloat Z;
    vpmfloat zeta0 = field->kernel.zeta(0.0f);
    vpmvec3 Uinf = field->Uinf;
    R relax = field->relaxation;

    // Reset temp variable
    particle.M = vpmmat3{ 0.0f };

    // Loop over the pairs
    for (int i = 0; i < 3; ++i) {
        vpmfloat a = rungeKuttaCoefs[i][0];
        vpmfloat b = rungeKuttaCoefs[i][1];

        // RUN SFS
        field->SFS(index, field, a, b);
        __syncthreads();

        particle.M[0] = a * particle.M[0] + dt * (particle.U + Uinf);
        particle.X += b * particle.M[0];

        S = xDotNablaY(particle.Gamma, particle.J);
        Z = 0.2f * glm::dot(S, particle.Gamma) / glm::dot(particle.Gamma, particle.Gamma);

        particle.M[1] = a * particle.M[1] + dt * (S - 3 * Z * particle.Gamma -
            particle.C[0] * particle.SFS * particle.sigma * particle.sigma * particle.sigma / zeta0);
        particle.M[2][1] = a * particle.M[2][1] - dt * (particle.sigma * Z);

        particle.Gamma += b * particle.M[1];
        particle.sigma += b * particle.M[2][1];

#ifdef SHARED_MEMORY
        // Copy variables back to global memory
        field->particles[index].Gamma = particle.Gamma;
        field->particles[index].sigma = particle.sigma;
        field->particles[index].X     = particle.X;
#endif
        __syncthreads();
    }

    if (useRelax) {
        particle.reset();
        calcVelJacNaive(index, field);

        __syncthreads(); // useRelax is the same for all threads

        relax(particle);

#ifdef SHARED_MEMORY
        // Copy variables back to global memory
        field->particles[index].Gamma = particle.Gamma;
#endif
    }
}

void writeVTK(int numParticles, Particle* particleBuffer, std::string filename, int timestep) {
    const int dim = 3;

    leanvtk::VTUWriter writer;

    std::vector<double> particleX;
    std::vector<double> particleU;
    std::vector<double> particleGamma;
    std::vector<double> particleOmega;
    std::vector<double> particleSigma;
    std::vector<double> particleIdx;
    particleX.reserve(dim * numParticles);
    particleU.reserve(dim * numParticles);
    particleGamma.reserve(dim * numParticles);
    particleOmega.reserve(dim * numParticles);
    particleSigma.reserve(numParticles);
    particleIdx.reserve(numParticles);

    vpmvec3 omega;

    for (int i = 0; i < numParticles; ++i) {
        Particle& particle = particleBuffer[i];

        particleIdx.push_back(i);
        particleSigma.push_back(particle.sigma);

        omega = nablaCrossX(particle.J);

        for (int j = 0; j < dim; ++j) {
            particleU.push_back(particle.U[j]);
            particleX.push_back(particle.X[j]);
            particleGamma.push_back(particle.Gamma[j]);
            particleOmega.push_back(omega[j]);
        }
    }

    writer.add_scalar_field("index", particleIdx);
    writer.add_scalar_field("sigma", particleSigma);
    writer.add_vector_field("position", particleX, dim);
    writer.add_vector_field("velocity", particleU, dim);
    writer.add_vector_field("circulation", particleGamma, dim);
    writer.add_vector_field("vorticity", particleOmega, dim);
    writer.write_point_cloud("../output/" + filename + "_" + std::to_string(timestep) + ".vtu", dim, particleX);
}

template <typename R, typename S, typename K>
void runVPM(
    int maxParticles,
    int numParticles,
    int numTimeSteps,
    vpmfloat dt,
    int fileSaveSteps,
    int blockSize,
    vpmvec3 uInf,
    Particle* particleBuffer,
    R relaxation,
    S sfs,
    K kernel,
    std::string filename) {
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

    writeVTK(numParticles, particleBuffer, filename, 0);

    for (int i = 1; i <= numTimeSteps; ++i) {
#ifdef SHARED_MEMORY
        rungekutta<R, S, K><<<fullBlocksPerGrid, blockSize, blockSize * sizeof(Particle)>>>(
            numParticles, dev_field, dt, true
        );
#else
        rungekutta<R, S, K><<<fullBlocksPerGrid, blockSize>>>(
            numParticles, dev_field, dt, true
        );
#endif

        if (i % fileSaveSteps == 0) {
            cudaMemcpy(particleBuffer, dev_particleBuffer, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
            //writeVTK(numParticles, particleBuffer, filename, i / fileSaveSteps);
            std::cout << particleBuffer[0].U.x << std::endl;
        }
    }

    // free device memory
    cudaFree(dev_particleBuffer);
    cudaFree(dev_field);
}

void randomCubeInit(Particle* particleBuffer, int N, vpmfloat cubeSize, vpmfloat maxCirculation, vpmfloat maxSigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<vpmfloat> uniform(-1, 1);
    std::uniform_real_distribution<vpmfloat> uniformPos(0, 1);

    for (int i = 0; i < N; ++i) {
        Particle& particle = particleBuffer[i];

        particle.sigma = maxSigma * uniformPos(gen);
        particle.Gamma = maxCirculation * uniform(gen) * glm::normalize(vpmvec3{ uniform(gen), uniform(gen), uniform(gen) });
        particle.circulation = glm::length(particle.Gamma);

        particle.X = cubeSize * uniform(gen) * glm::normalize(vpmvec3{ uniform(gen), uniform(gen), uniform(gen) });
    }
}

void randomSphereInit(Particle* particleBuffer, int N, vpmfloat sphereRadius, vpmfloat maxCirculation, vpmfloat maxSigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<vpmfloat> uniform(-1, 1);
    std::uniform_real_distribution<vpmfloat> uniformPos(0, 1);

    for (int i = 0; i < N; ++i) {
        Particle& particle = particleBuffer[i];

        particle.sigma = maxSigma * uniformPos(gen);
        particle.Gamma = maxCirculation * uniform(gen) * glm::normalize(vpmvec3{ uniform(gen), uniform(gen), uniform(gen) });
        particle.circulation = glm::length(particle.Gamma);

        vpmfloat theta = 2 * PI * uniformPos(gen);
        vpmfloat phi = PI * uniformPos(gen);
        vpmfloat radius = std::cbrt(uniformPos(gen)) * sphereRadius;

        vpmfloat x = radius * sin(phi) * cos(theta);
        vpmfloat y = radius * sin(phi) * sin(theta);
        vpmfloat z = radius * cos(phi);
        particle.X = vpmvec3{ x, y, z };
    }
}

void runSimulation() {
    // Define basic parameters
    int maxParticles{ 6000 };
    int numTimeSteps{ 10 };
    vpmfloat dt{ 0.01f };
    int numBlocks{ 128 };
    int numStepsVTK{ 1 };
    vpmvec3 uInf{ 0, 0, 0 };

    // Create host particle buffer
    Particle* particleBuffer = new Particle[maxParticles];
    // Initialize particle buffer
    //randomSphereInit(particleBuffer, maxParticles, 10.0f, 1.0f, 0.5f);
    int numParticles = initVortexRings(particleBuffer, maxParticles);

    // Run VPM method
    runVPM(
        maxParticles,
        numParticles,
        numTimeSteps,
        dt,
        numStepsVTK,
        numBlocks,
        uInf,
        particleBuffer,
        CorrectedPedrizzettiRelaxation(0.3f),
        NoSFS(),
        GaussianErfKernel(),
        "test"
    );

    // Free host particle buffer
    delete[] particleBuffer;
}