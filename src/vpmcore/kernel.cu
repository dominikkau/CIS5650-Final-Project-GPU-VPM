#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include "kernel.h"
#include "../lean_vtk.hpp"
#include "../vortexringsimulation.hpp"
#include <device_launch_parameters.h>

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#ifdef ENABLE_CUDA_ERROR
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
#endif
}

// Constructor
Particle::Particle() 
    : X(0.0f), Gamma(0.0f), sigma(0.0f), vol(0.0f), circulation(0.0f), isStatic(false),
      U(0.0f), J(0.0f), PSE(0.0f), M(0.0f), C(0.0f), SFS(0.0f) {}

__host__ __device__ void Particle::reset() {
    U   = vpmvec3(0.0f);
    J   = vpmmat3(0.0f);
    PSE = vpmvec3(0.0f);
}

__host__ __device__ void Particle::resetSFS() {
    SFS = vpmvec3(0.0f);
}

// *************************************************************
// *                      RELAXATION                           *
// *************************************************************

inline void PedrizzettiRelaxation::operator()(int N, Particle* particles, int numBlocks, int blockSize) {
    pedrizzettiRelax<<<numBlocks, blockSize>>>(N, particles, relaxFactor);
    cudaDeviceSynchronize();
    checkCUDAError("pedrizzettiRelax failed!");
}

__global__ void pedrizzettiRelax(int N, Particle* particles, vpmfloat relaxFactor) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3 omega = nablaCrossX(particles[index].J);
    const vpmvec3 oldGamma = particles[index].Gamma;

    particles[index].Gamma = (1.0f - relaxFactor) * oldGamma
                     + relaxFactor * glm::length(oldGamma) / glm::length(omega) * omega;
}

inline void CorrectedPedrizzettiRelaxation::operator()(int N, Particle* particles, int numBlocks, int blockSize) {
    correctedPedrizzettiRelax<<<numBlocks, blockSize>>>(N, particles, relaxFactor);
    cudaDeviceSynchronize();
    checkCUDAError("correctedPedrizzettiRelax failed!");
}

__global__ void correctedPedrizzettiRelax(int N, Particle* particles, vpmfloat relaxFactor) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3  omega = nablaCrossX(particles[index].J);
    const vpmvec3  oldGamma = particles[index].Gamma;
    const vpmfloat omegaNorm = glm::length(omega);
    const vpmfloat gammaNorm = glm::length(oldGamma);

    const vpmfloat tmp = sqrt(1.0 - 2.0 * (1.0 - relaxFactor) * relaxFactor
        * (1.0 - glm::dot(oldGamma, omega) / (omegaNorm * gammaNorm)));

    particles[index].Gamma = ((1.0f - relaxFactor) * oldGamma
        + relaxFactor * gammaNorm / omegaNorm * omega) / tmp;
}

// *************************************************************
// *                     SFS modeling                          *
// *************************************************************

__global__ void calculateTemporary(int N, Particle* particles, bool testFilter) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    Particle& particle = particles[index];

    if (testFilter) {
        particle.M[0] = xDotNablaY(particle.Gamma, particle.J);
        particle.M[1] = particle.SFS;
    }
    else {
        particle.M[0] -= xDotNablaY(particle.Gamma, particle.J);
        particle.M[1] -= particle.SFS;
    }
}

__global__ void calculateCoefficient(int N, Particle* particles, vpmfloat zeta0,
    vpmfloat alpha, vpmfloat relaxFactor, bool forcePositive, vpmfloat minC, vpmfloat maxC) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3 particleGamma = particles[index].Gamma;
    const vpmvec3 particleSFS = particles[index].SFS;
    const vpmmat3 particleM = particles[index].M;
    const vpmfloat particleSigma = particles[index].sigma;

    vpmvec3 particleC = particles[index].C;

    vpmfloat numerator = glm::dot(particleM[0], particleGamma);
    numerator *= 3.0 * alpha - 2.0;

    vpmfloat denominator = glm::dot(particleM[1], particleGamma);
    denominator *= particleSigma * particleSigma * particleSigma / zeta0;

    // Don't initialize denominator to 0
    if (particleC[2] == 0) particleC[2] = denominator;

    // Lagrangian average
    numerator = relaxFactor * numerator + (1 - relaxFactor) * particleC[1];
    denominator = relaxFactor * denominator + (1 - relaxFactor) * particleC[2];

    // Enforce maximum and minimum absolute values
    if (fabs(numerator / denominator) > maxC) {
        if (fabs(denominator) < fabs(particleC[2])) denominator = copysign(particleC[2], denominator);

        if (fabs(numerator / denominator) > maxC) numerator = copysign(denominator, numerator) * maxC;
    }
    else if (fabs(numerator / denominator) < minC) numerator = copysign(denominator, numerator) * minC;

    // Save numerator and denominator of model coefficient
    particleC[1] = numerator;
    particleC[2] = denominator;

    // Store model coefficient
    particleC[0] = particleC[1] / particleC[2];

    // Force the coefficient to be positive
    if (forcePositive) particleC[0] = fabs(particleC[0]);

    // Clipping
    if (particleC[0] * glm::dot(particleGamma, particleSFS) < 0) particleC[0] = 0;

    // Copy result to global memory
    particles[index].C = particleC;
    particles[index].M = vpmmat3{ 0.0 };
}

template <typename R, typename S, typename K>
void DynamicSFS::operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize) {
    if (a == 1.0 || a == 0.0) {
        K& kernel = field.kernel;
        Particle* particles = field.particles;
        int N = field.np;

        // CALCULATIONS WITH TEST FILTER
        calcVelJacNaive<<<numBlocks, blockSize>>>(N, N, particles, particles, kernel, true, alpha);
        cudaDeviceSynchronize();
        checkCUDAError("calcVelJacNaive (DynamicsSFS: test filter) failed!");

        calcEstrNaive<<<numBlocks, blockSize>>>(N, N, particles, particles, kernel, true, alpha);
        cudaDeviceSynchronize();
        checkCUDAError("calcEstrNaive (DynamicsSFS: test filter) failed!");

        calculateTemporary<<<numBlocks, blockSize>>>(N, particles, true);
        cudaDeviceSynchronize();
        checkCUDAError("calculateTemporary (DynamicsSFS: test filter) failed!");

        // CALCULATIONS WITH DOMAIN FILTER
        calcVelJacNaive<<<numBlocks, blockSize>>>(N, N, particles, particles, kernel, true);
        cudaDeviceSynchronize();
        checkCUDAError("calcVelJacNaive (DynamicsSFS: domain filter) failed!");

        calcEstrNaive<<<numBlocks, blockSize>>>(N, N, particles, particles, kernel, true);
        cudaDeviceSynchronize();
        checkCUDAError("calcEstrNaive (DynamicsSFS: domain filter) failed!");

        calculateTemporary<<<numBlocks, blockSize>>>(N, particles, false);
        cudaDeviceSynchronize();
        checkCUDAError("calculateTemporary (DynamicsSFS: domain filter) failed!");

        // CALCULATE COEFFICIENT
        calculateCoefficient<<<numBlocks, blockSize>>>(N, particles, kernel.zeta(0.0), alpha,
            relaxFactor, forcePositive, minC, maxC);
        cudaDeviceSynchronize();
        checkCUDAError("calculateCoefficient failed!");
    }
    else {
        calcVelJacNaive<<<numBlocks, blockSize>>>(N, N, particles, particles, kernel, true);
        cudaDeviceSynchronize();
        checkCUDAError("calcVelJacNaive (DynamicsSFS: 2nd step) failed!");

        calcEstrNaive<<<numBlocks, blockSize>>>(N, N, particles, particles, kernel, true);
        cudaDeviceSynchronize();
        checkCUDAError("calcEstrNaive (DynamicsSFS: 2nd step) failed!");
    }
}

template <typename R, typename S, typename K>
void NoSFS::operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize) {
    resetParticlesSFS<<<numBlocks, blockSize>>>(field.np, field.particles);
    cudaDeviceSynchronize();
    checkCUDAError("resetParticlesSFS (NoSFS) failed!");

    calcVelJacNaive<<<numBlocks, blockSize >>>(field.np, field.np, field.particles, field.particles, field.kernel, true);
    cudaDeviceSynchronize();
    checkCUDAError("calcVelJacNaive (NoSFS) failed!");
}

__global__ void resetParticles(int N, Particle* particles) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    particles[index].U = vpmvec3{ 0.0 };
    particles[index].J = vpmmat3{ 0.0 };
    particles[index].PSE = vpmvec3{ 0.0 };
}

__global__ void resetParticlesSFS(int N, Particle* particles) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    particles[index].SFS = vpmvec3{ 0.0 };
}

template <typename K>
__global__ void calcEstrNaive(int targetN, int sourceN, Particle* targetParticles,
    Particle* sourceParticles, K kernel, bool reset, vpmfloat testFilterFactor) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= targetN) return;

    // Get required variables from global memory
    const vpmvec3 targetX = targetParticles[index].X;
    const vpmmat3 targetJ = targetParticles[index].J;
    vpmvec3 targetSFS;
    if (reset) {
        targetSFS = vpmvec3{ 0.0 };
    }
    else {
        targetSFS = targetParticles[index].SFS;
    }

    for (int i = 0; i < sourceN; ++i) {
        Particle& sourceParticle = sourceParticles[i];
        const vpmfloat sourceSigma = sourceParticle.sigma;

        targetSFS += kernel.zeta(glm::length(targetX - sourceParticle.X) / sourceSigma)
            / (sourceSigma * sourceSigma * sourceSigma)
            * xDotNablaY(sourceParticle.Gamma, targetJ - sourceParticle.J);
    }

    // Copy variables back to global memory
    targetParticles[index].SFS = targetSFS;
}

template <typename K>
__global__ void calcVelJacNaive(int targetN, int sourceN, Particle* targetParticles, 
    Particle* sourceParticles, K kernel, bool reset, vpmfloat testFilterFactor) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= targetN) return;

    // Get required variables from global memory
    const vpmvec3 targetX = targetParticles[index].X;

    vpmvec3 targetU;
    vpmmat3 targetJ;
    if (reset) {
        targetU = vpmvec3{ 0.0 };
        targetJ = vpmmat3{ 0.0 };
    }
    else {
        targetU = targetParticles[index].U;
        targetJ = targetParticles[index].J;
    }
    
    for (int i = 0; i < sourceN; ++i) {
        if (i == index) continue;

        Particle& sourceParticle = sourceParticles[i];
        const vpmfloat invSourceSigma = 1 / (sourceParticle.sigma * testFilterFactor);
        const vpmvec3 sourceGamma = sourceParticle.Gamma;

        const vpmvec3 dX = targetX - sourceParticle.X;
        const vpmfloat r = glm::length(dX);

        // is this needed?
        if (r < EPS) continue;

        // Kernel evaluation
        const vpmfloat g_sgm = kernel.g(r * invSourceSigma);
        const vpmfloat dg_sgmdr = kernel.dgdr(r * invSourceSigma);

        // Compute velocity
        const vpmvec3 crossProd = glm::cross(dX, sourceGamma) * (-const4 / (r*r*r));
        targetU += g_sgm * crossProd;

        // Compute Jacobian
        vpmfloat tmp = dg_sgmdr * invSourceSigma / r - 3.0 * g_sgm / (r*r);
        const vpmvec3 dX_norm = dX / r;

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                targetJ[l][k] += tmp * crossProd[k] * dX_norm[l];
            }
        }

        tmp = - const4 * g_sgm / (r*r*r);

        // Account for kronecker delta term
        targetJ[0][1] -= tmp * sourceGamma[2];
        targetJ[0][2] += tmp * sourceGamma[1];
        targetJ[1][0] += tmp * sourceGamma[2];
        targetJ[1][2] -= tmp * sourceGamma[0];
        targetJ[2][0] -= tmp * sourceGamma[1];
        targetJ[2][1] += tmp * sourceGamma[0];
    }

    // Copy variables back to global memory
    targetParticles[index].U = targetU;
    targetParticles[index].J = targetJ;
}

__global__ void rungeKuttaStep(int N, Particle* particles, vpmfloat a, vpmfloat b, vpmfloat dt, vpmfloat zeta0, vpmvec3 Uinf) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmfloat particleC = particles[index].C[0];
    const vpmvec3 particleU = particles[index].U;
    const vpmvec3 particleSFS = particles[index].SFS;
    const vpmmat3 particleJ = particles[index].J;
    
    vpmfloat particleSigma = particles[index].sigma;
    vpmvec3 particleGamma = particles[index].Gamma;
    vpmvec3 particleX = particles[index].X;
    vpmmat3 particleM;
    if (a == 1.0 || a == 0.0) {
        particleM = vpmmat3{ 0.0 };
    }
    else {
        particleM = particles[index].M;
    }
    
    particleM[0] = a * particleM[0] + dt * (particleU + Uinf);
    particleX += b * particleM[0];

    vpmvec3 S = xDotNablaY(particleGamma, particleJ);
    vpmfloat Z = 0.2 * glm::dot(S, particleGamma) / glm::dot(particleGamma, particleGamma);

    particleM[1] = a * particleM[1] + dt * (S - 3 * Z * particleGamma -
        particleC * particleSFS * particleSigma * particleSigma * particleSigma / zeta0);
    particleM[2][1] = a * particleM[2][1] - dt * (particleSigma * Z);

    particleGamma += b * particleM[1];
    particleSigma += b * particleM[2][1];

    // Copy variables back to global memory
    particles[index].X = particleX;
    particles[index].Gamma = particleGamma;
    particles[index].sigma = particleSigma;
    particles[index].M = particleM;
}

template <typename R, typename S, typename K>
void rungeKutta(ParticleField<R, S, K>& field, vpmfloat dt, bool useRelax, int numBlocks, int blockSize) {

    const vpmfloat rungeKuttaCoefs[3][2] = {
        {0.0, 1.0 / 3.0},
        {-5.0 / 9.0, 15.0 / 16.0},
        {-153.0 / 128.0, 8.0 / 15.0}
    };

    K kernel = field.kernel;
    R relax = field.relaxation;

    // Loop over the pairs
    for (int i = 0; i < 3; ++i) {
        vpmfloat a = rungeKuttaCoefs[i][0];
        vpmfloat b = rungeKuttaCoefs[i][1];

        // RUN SFS
        field.SFS(field, a, b, numBlocks, blockSize);

        rungeKuttaStep<<<numBlocks, blockSize>>>(field.np, field.particles, a, b, dt, kernel.zeta(0.0), field.Uinf);
        cudaDeviceSynchronize();
        checkCUDAError("rungeKuttaStep failed!");
    }

    if (useRelax) {
        calcVelJacNaive<<<numBlocks, blockSize>>>(field.np, field.np, field.particles, field.particles, kernel, true);
        cudaDeviceSynchronize();
        checkCUDAError("calcVelJacNaive (rungeKutta: Relaxation) failed!");

        relax(field.np, field.particles, numBlocks, blockSize);
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
    vpmvec3 uInf,
    Particle* particleBuffer,
    R relaxation,
    S sfs,
    K kernel,
    int blockSize,
    std::string filename) {

    int numBlocks{ (numParticles + blockSize - 1) / blockSize };

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

    std::cout << particleBuffer[0].U.x << std::endl;

    writeVTK(numParticles, particleBuffer, filename, 0);

    for (int i = 1; i <= numTimeSteps; ++i) {
        rungeKutta(field, dt, true, numBlocks, blockSize);

        if (i % fileSaveSteps == 0) {
            cudaMemcpy(particleBuffer, dev_particleBuffer, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
            
            writeVTK(numParticles, particleBuffer, filename, i / fileSaveSteps);
            std::cout << particleBuffer[0].U.x << std::endl;
        }
    }

    // free device memory
    cudaFree(dev_particleBuffer);
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
    int maxParticles{ 50000 };
    int numTimeSteps{ 2000 };
    vpmfloat dt{ 0.01f };
    int numStepsVTK{ 1 };
    vpmvec3 uInf{ 0, 0, 0 };
    int blockSize{ 128 };

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
        uInf,
        particleBuffer,
        PedrizzettiRelaxation(0.3),
        NoSFS(),
        WinckelmansKernel(),
        blockSize,
        "test"
    );

    // Free host particle buffer
    delete[] particleBuffer;
}