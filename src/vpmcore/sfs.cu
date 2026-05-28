#include "common.h"
#include "sfs.h"
#include "kernels.h"
#include "vpmmain.h"

__global__ void calculateTemporary(int N, ParticleBuffer particles, bool testFilter) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    if (testFilter) {
        particles.M()[index][0] = xDotNablaY(particles.Gamma()[index], particles.J()[index]);
        particles.M()[index][1] = particles.SFS()[index];
    }
    else {
        particles.M()[index][0] -= xDotNablaY(particles.Gamma()[index], particles.J()[index]);
        particles.M()[index][1] -= particles.SFS()[index];
    }
}

__global__ void calculateCoefficient(int N, ParticleBuffer particles, vpm::real zeta0,
    vpm::real alpha, vpm::real relaxFactor, bool forcePositive, vpm::real minC, vpm::real maxC) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpm::vec3 particleGamma  = particles.Gamma()[index];
    const vpm::vec3 particleSFS    = particles.SFS()[index];
    const vpm::mat3 particleM      = particles.M()[index];
    const vpm::real particleSigma = particles.sigma()[index];

    vpm::vec3 particleC = particles.C()[index];

    vpm::real numerator = glm::dot(particleM[0], particleGamma);
    numerator *= 3.0f * alpha - 2.0f;

    vpm::real denominator = glm::dot(particleM[1], particleGamma);
    denominator *= particleSigma * particleSigma * particleSigma / zeta0;

    // Don't initialize denominator to 0
    if (particleC[2] == 0) particleC[2] = denominator;

    // Lagrangian average
    numerator = relaxFactor * numerator + (1.0f - relaxFactor) * particleC[1];
    denominator = relaxFactor * denominator + (1.0f - relaxFactor) * particleC[2];

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
    particles.C()[index] = particleC;
}

void DynamicSFS::operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream) {
    KernelType kernel = field.kernel;
    ParticleBuffer& particles = field.dev_particles;
    const int N = field.numParticles;
    const CUDAKernelParams velParams{ numBlocks, blockSize, 7 * blockSize * sizeof(vpm::real), stream };
    const CUDAKernelParams estrParams{ numBlocks, blockSize, 16 * blockSize * sizeof(vpm::real), stream };

    if (a == 1.0f || a == 0.0f) {
        // CALCULATIONS WITH TEST FILTER
        calcVelJacNaiveWrapper(velParams, N, N, particles, particles, kernel, true, alpha);
        checkCUDAError("calcVelJacNaive (DynamicsSFS: test filter) failed!");

        calcEstrNaiveWrapper(estrParams, N, N, particles, particles, kernel, true, alpha);
        checkCUDAError("calcEstrNaive (DynamicsSFS: test filter) failed!");

        calculateTemporary<<<numBlocks, blockSize, 0, stream>>>(N, particles, true);
        checkCUDAError("calculateTemporary (DynamicsSFS: test filter) failed!");

        // CALCULATIONS WITH DOMAIN FILTER
        calcVelJacNaiveWrapper(velParams, N, N, particles, particles, kernel, true);
        checkCUDAError("calcVelJacNaive (DynamicsSFS: domain filter) failed!");

        calcEstrNaiveWrapper(estrParams, N, N, particles, particles, kernel, true);
        checkCUDAError("calcEstrNaive (DynamicsSFS: domain filter) failed!");

        calculateTemporary<<<numBlocks, blockSize, 0, stream>>>(N, particles, false);
        checkCUDAError("calculateTemporary (DynamicsSFS: domain filter) failed!");

        // CALCULATE COEFFICIENT
        const Kernel* kernelPointer = getKernel(kernel);
        calculateCoefficient<<<numBlocks, blockSize, 0, stream>>>(N, particles, kernelPointer->zeta(0.0), alpha,
            relaxFactor, forcePositive, minC, maxC);
        checkCUDAError("calculateCoefficient failed!");
        delete kernelPointer;
    }
    else {
        calcVelJacNaiveWrapper(velParams, N, N, particles, particles, kernel, true);
        checkCUDAError("calcVelJacNaive (DynamicsSFS: 2nd step) failed!");

        calcEstrNaiveWrapper(estrParams, N, N, particles, particles, kernel, true);
        checkCUDAError("calcEstrNaive (DynamicsSFS: 2nd step) failed!");
    }
}

void NoSFS::operator()(ParticleField& field, vpm::real a, vpm::real b, int numBlocks, int blockSize, cudaStream_t stream) {
    const int N = field.numParticles;
    const CUDAKernelParams params{ numBlocks, blockSize, 7 * blockSize * sizeof(vpm::real), stream };

    cudaMemset(field.dev_particles.SFS(), 0, N * sizeof(vpm::vec3));
    checkCUDAError("cudaMemset (SFS reset) failed!");

    calcVelJacNaiveWrapper(params, N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (NoSFS) failed!");
}