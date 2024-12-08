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
// *            PARTICLE FIELD IMPLEMENTATION                  *
// *************************************************************

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::addParticle(Particle& particle) {
    if (numParticles == maxParticles) return;

    cudaMemcpy(particles[numParticles], &particle, sizeof(Particle), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkCUDAError("cudaMemcpy (addParticle) failed!");

    ++numParticles;
}

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::removeParticle(int index) {
    // not the last particle
    if (index != numParticles - 1) {
        cudaMemcpy(particles[index], particles[numParticles], sizeof(Particle), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        checkCUDAError("cudaMemcpy (removeParticle) failed!");
    }

    --numParticles;
}

template <typename R, typename S, typename K>
ParticleField<R, S, K>::ParticleField(
    int maxParticles,
    Particle* particles,
    int numParticles,
    int timeSteps,
    K kernel,
    vpmvec3 uInf,
    S sfs,
    R relaxation)
    :
    maxParticles(maxParticles),
    particles(particles),
    numParticles(numParticles),
    timeStep(timeStep),
    kernel(kernel),
    uInf(uInf),
    sfs(sfs),
    relaxation(relaxation),
    synchronized(true) {

    // Declare device particle buffer
    cudaMalloc((void**)&dev_particles, maxParticles * sizeof(Particle));
    cudaDeviceSynchronize();
    checkCUDAError("cudaMalloc of dev_particleBuffer failed!");

    // Copy particle buffer from host to device
    cudaMemcpy(dev_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkCUDAError("cudaMemcpy particleBuffer->dev_particleBuffer failed!");
};

template <typename R, typename S, typename K>
ParticleField<R, S, K>::~ParticleField() {
    // free device memory
    cudaFree(dev_particles);
}

template <typename R, typename S, typename K>
Particle* ParticleField<R, S, K>::getParticles() {
    if (!synchronized) {
        cudaMemcpy(particles, dev_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        checkCUDAError("cudaMemcpy dev_particleBuffer->particleBuffer failed!");
        synchronized = true;
    }

    return particles;
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

    const vpmvec3 omega    = nablaCrossX(particles[index].J);
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

    const vpmvec3 omega      = nablaCrossX(particles[index].J);
    const vpmvec3 oldGamma   = particles[index].Gamma;
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

    const vpmvec3 particleGamma  = particles[index].Gamma;
    const vpmvec3 particleSFS    = particles[index].SFS;
    const vpmmat3 particleM      = particles[index].M;
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
}

template <typename R, typename S, typename K>
void DynamicSFS::operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize) {
    K& kernel = field.kernel;
    Particle* particles = field.dev_particles;
    const int N = field.numParticles;

    if (a == 1.0 || a == 0.0) {
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
    const int N = field.numParticles;

    resetParticlesSFS<<<numBlocks, blockSize>>>(N, field.dev_particles);
    cudaDeviceSynchronize();
    checkCUDAError("resetParticlesSFS (NoSFS) failed!");

    calcVelJacNaive<<<numBlocks, blockSize>>>(N, N, field.dev_particles, field.dev_particles, field.kernel, true);
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
        const vpmfloat invSourceSigma = 1 / (sourceParticle.sigma * testFilterFactor);

        targetSFS += kernel.zeta(glm::length(targetX - sourceParticle.X) * invSourceSigma)
            * invSourceSigma * invSourceSigma * invSourceSigma
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
        Particle& sourceParticle = sourceParticles[i];
        const vpmfloat invSourceSigma = 1 / (sourceParticle.sigma * testFilterFactor);
        const vpmvec3 sourceGamma = sourceParticle.Gamma;

        const vpmvec3 dX = targetX - sourceParticle.X;
        const vpmfloat r = glm::length(dX);

        // is this needed?
        if (r == 0.0) continue;

        // Kernel evaluation
        const vpmfloat g_sgm = kernel.g(r * invSourceSigma);
        const vpmfloat dg_sgmdr = kernel.dgdr(r * invSourceSigma);

        // Compute velocity
        const vpmvec3 crossProd = -const4 / (r * r * r) * glm::cross(dX, sourceGamma);
        targetU += g_sgm * crossProd;
        
        // Compute Jacobian
        vpmfloat tmp = dg_sgmdr * invSourceSigma / r - 3.0 * g_sgm / (r * r);

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                targetJ[l][k] += tmp * crossProd[k] * dX[l];
            }
        }

        tmp = - const4 * g_sgm / (r * r * r);

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

    const vpmfloat particleC   = particles[index].C[0];
    const vpmvec3  particleU   = particles[index].U;
    const vpmvec3  particleSFS = particles[index].SFS;
    const vpmmat3  particleJ   = particles[index].J;
    
    vpmfloat particleSigma = particles[index].sigma;
    vpmvec3  particleGamma = particles[index].Gamma;
    vpmvec3  particleX     = particles[index].X;
    vpmmat3  particleM;
    if (a == 1.0 || a == 0.0) {
        particleM = vpmmat3{ 0.0 };
    }
    else {
        particleM = particles[index].M;
    }
    
    // Position update
    particleM[0] = a * particleM[0] + dt * (particleU + Uinf);
    particleX += b * particleM[0];
    particles[index].X = particleX;

    vpmvec3 S = xDotNablaY(particleGamma, particleJ);
#ifdef CLASSIC_VPM
    vpmfloat Z = 0.0;
#else
    vpmfloat Z = 0.2 * glm::dot(S, particleGamma) / glm::dot(particleGamma, particleGamma);
#endif

    // Gamma update
    particleM[1] = a * particleM[1] + dt * (S - 3 * Z * particleGamma
        - particleC * particleSFS * particleSigma * particleSigma * particleSigma / zeta0);
    particleGamma += b * particleM[1];
    particles[index].Gamma = particleGamma;

#ifndef CLASSIC_VPM
    // Sigma update
    particleM[2][1] = a * particleM[2][1] - dt * (particleSigma * Z);
    particleSigma += b * particleM[2][1];

    particles[index].sigma = particleSigma;
#endif

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
    const int N = field.numParticles;

    // Loop over the pairs
    for (int i = 0; i < 3; ++i) {
        vpmfloat a = rungeKuttaCoefs[i][0];
        vpmfloat b = rungeKuttaCoefs[i][1];

        // RUN SFS
        field.sfs(field, a, b, numBlocks, blockSize);

        rungeKuttaStep<<<numBlocks, blockSize>>>(N, field.dev_particles, a, b, dt, kernel.zeta(0.0), field.uInf);
        cudaDeviceSynchronize();
        checkCUDAError("rungeKuttaStep failed!");
    }

    if (useRelax) {
        calcVelJacNaive<<<numBlocks, blockSize>>>(N, N, field.dev_particles, field.dev_particles, kernel, true);
        cudaDeviceSynchronize();
        checkCUDAError("calcVelJacNaive (rungeKutta: Relaxation) failed!");

        field.relaxation(N, field.dev_particles, numBlocks, blockSize);
    }

    ++field.timeStep;
    field.synchronized = false;
}

template <typename R, typename S, typename K>
void writeVTK(const ParticleField<R, S, K>& field, const std::string filename) {
    const int dim = 3;
    const Particle* particleBuffer = field.getParticles();

    static leanvtk::VTUWriter writer;

    static std::vector<double> particleX;
    static std::vector<double> particleU;
    static std::vector<double> particleGamma;
    static std::vector<double> particleOmega;
    static std::vector<double> particleSigma;
    static std::vector<double> particleIdx;
    particleX.reserve(field.maxParticles * dim);
    particleU.reserve(field.maxParticles * dim);
    particleGamma.reserve(field.maxParticles * dim);
    particleOmega.reserve(field.maxParticles * dim);
    particleSigma.reserve(field.maxParticles);
    particleIdx.reserve(field.maxParticles);

    vpmvec3 omega;
    for (int i = 0; i < field.numParticles; ++i) {
        const Particle& particle = particleBuffer[i];

        particleIdx.push_back(particle.index);
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
    writer.write_point_cloud("../output/" + filename + "_" + std::to_string(field.timeStep) + ".vtu", dim, particleX);

    writer.clear();
    particleX.clear();
    particleU.clear();
    particleGamma.clear();
    particleOmega.clear();
    particleSigma.clear();
    particleIdx.clear();
}

template <typename R, typename S, typename K>
void calcVortexRingMetrics(ParticleField<R, S, K>& field, int iteration, std::string filename, int numRings = 2) {
    const Particle* particleBuffer = field.getParticles();
    int numParticlesRing = field.numParticles / numRings;

    std::vector<vpmfloat> ringRadii;
    std::vector<vpmvec3>  ringCenters;

    for (int j = 0; j < numRings; ++j) {
        int offset = j * numParticlesRing;
        // Calculate ring center
        vpmvec3 ringCenter = vpmvec3{ 0 };
        vpmfloat totalGamma = 0;
        for (int i = offset; i < numParticlesRing + offset; ++i) {
            vpmfloat Gamma = glm::length(particleBuffer[i].Gamma);
            totalGamma += Gamma;
            ringCenter += Gamma * particleBuffer[i].X;
        }
        ringCenter /= totalGamma;

        // Calculate ring radius
        vpmfloat ringRadius = 0;
        for (int i = offset; i < numParticlesRing + offset; ++i) {
            vpmfloat Gamma = glm::length(particleBuffer[i].Gamma);
            vpmfloat radius = glm::length(particleBuffer[i].X - ringCenter);
            ringRadius += Gamma * radius;
        }
        ringRadius /= totalGamma;

        ringCenters.push_back(ringCenter);
        ringRadii.push_back(ringRadius);
    }

    // Save results to csv file
    filename = "../output/" + filename + ".csv";
    std::ofstream file;
    if (iteration == 0) {
        // Overwrite file and write header in the first iteration
        file.open(filename, std::ios::out);
        if (file.is_open()) {
            file << "iteration";
            for (int i = 1; i < numRings + 1; ++i) {
                file << ",ring_center_" << i
                     << ",ring_radius_ " << i;
            }
            file << '\n';
        }
    }
    else {
        // Append to file in subsequent iterations
        file.open(filename, std::ios::app);
    }

    if (file.is_open()) {
        // Write iteration, ring center (Z-coordinate), and radius
        file << iteration;
        for (int i = 1; i < numRings + 1; ++i) {
            file << ',' << ringCenters[i-1][2]
                 << ',' << ringRadii[i-1];
        }
        file << '\n';
        file.close();
    }
    else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
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

    ParticleField<R, S, K> field{
        maxParticles,
        particleBuffer,
        numParticles,
        0,
        kernel,
        uInf,
        sfs,
        relaxation
    };


    for (int i = 0; i < numTimeSteps + 1; ++i) {
        calcVortexRingMetrics(field, i, "test");

        if ((fileSaveSteps != 0) && (i % fileSaveSteps == 0)) {
            writeVTK(field, filename);
            std::cout << field.particles[0].U[0] << std::endl;
        }

        rungeKutta(field, dt, true, numBlocks, blockSize);
    }
}

void randomCubeInit(Particle* particleBuffer, int N, vpmfloat cubeSize, vpmfloat maxCirculation, vpmfloat maxSigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<vpmfloat> uniform(-1, 1);
    std::uniform_real_distribution<vpmfloat> uniformPos(0, 1);

    for (int i = 0; i < N; ++i) {
        Particle& particle = particleBuffer[i];

        particle.index = i;
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

        particle.index = i;
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
    int maxParticles{ 2000 };
    int numTimeSteps{ 2000 };
    vpmfloat dt{ 0.01f };
    int numStepsVTK{ 5 };
    vpmvec3 uInf{ 0, 0, 0 };
    int blockSize{ 128 };

    // Create host particle buffer
    Particle* particleBuffer = new Particle[maxParticles];
    // Initialize particle buffer
    //randomSphereInit(particleBuffer, maxParticles, 10.0f, 1.0f, 0.5f);
    //int numParticles = maxParticles;
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
        CorrectedPedrizzettiRelaxation(0.3),
        NoSFS(),
        WinckelmansKernel(),
        blockSize,
        "test"
    );

    // Free host particle buffer
    delete[] particleBuffer;
}