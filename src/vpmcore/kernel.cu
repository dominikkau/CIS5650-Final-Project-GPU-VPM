#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <random>
#include <memory>
#include "kernel.h"
#include "../lean_vtk.hpp"
#include "../vortexringsimulation.hpp"
#include "../roundjetsimulation.hpp"
#include <device_launch_parameters.h>

__host__ __device__ void Particle::reset() {
    U   = vpmvec3{ 0.0f };
    J   = vpmmat3{ 0.0f };
    //PSE = vpmvec3{ 0.0f };
}

__host__ __device__ void Particle::resetSFS() {
    SFS = vpmvec3{ 0.0f };
}

// *************************************************************
// *            PARTICLE FIELD IMPLEMENTATION                  *
// *************************************************************

struct ParticleBuffer;

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::cpyParticlesDeviceToDevice(ParticleBuffer inParticles, unsigned int inNumParticles, 
    unsigned int startIndex, int bufferMask) {

    numParticles = cpyParticleBuffer(dev_particles, inParticles, numParticles,
        maxParticles, inNumParticles, startIndex, bufferMask);
}

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::addParticleDevice(Particle& particle) {
    if (numParticles == maxParticles) return;

    Particle* dev_tmpParticle;
	cudaMalloc((void**)&dev_tmpParticle, sizeof(Particle));
	checkCUDAError("cudaMalloc of dev_tmpParticle failed!");

	cudaMemcpy(dev_tmpParticle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy of dev_tmpParticle failed!");

	cudaMemcpy(dev_particles.X + numParticles, &dev_tmpParticle->X, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.U + numParticles, &dev_tmpParticle->U, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.J + numParticles, &dev_tmpParticle->J, sizeof(vpmmat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.Gamma + numParticles, &dev_tmpParticle->Gamma, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.sigma + numParticles, &dev_tmpParticle->sigma, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.SFS + numParticles, &dev_tmpParticle->SFS, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.C + numParticles, &dev_tmpParticle->C, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.M + numParticles, &dev_tmpParticle->M, sizeof(vpmmat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.index + numParticles, &dev_tmpParticle->index, sizeof(int), cudaMemcpyDeviceToDevice);
    /*cudaMemcpy(dev_particles.PSE + numParticles, &dev_tmpParticle->PSE, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.isStatic + numParticles, &dev_tmpParticle->isStatic, sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.vol + numParticles, &dev_tmpParticle->vol, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.circulation + numParticles, &dev_tmpParticle->circulation, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);*/

    ++numParticles;

	cudaFree(dev_tmpParticle);
}

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::overwriteParticleDevice(Particle& particle, unsigned int index) {
    if (index > numParticles) {
        addParticle(particle);
        return;
    }

    Particle* dev_tmpParticle;
    cudaMalloc((void**)&dev_tmpParticle, sizeof(Particle));
    checkCUDAError("cudaMalloc of dev_tmpParticle failed!");

    cudaMemcpy(dev_tmpParticle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy of dev_tmpParticle failed!");

    cudaMemcpy(dev_particles.X + index, &dev_tmpParticle->X, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.U + index, &dev_tmpParticle->U, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.J + index, &dev_tmpParticle->J, sizeof(vpmmat3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.Gamma + index, &dev_tmpParticle->Gamma, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.sigma + index, &dev_tmpParticle->sigma, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.SFS + index, &dev_tmpParticle->SFS, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.C + index, &dev_tmpParticle->C, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.M + index, &dev_tmpParticle->M, sizeof(vpmmat3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.index + index, &dev_tmpParticle->index, sizeof(int), cudaMemcpyDeviceToDevice);
    /*cudaMemcpy(dev_particles.PSE + index, &dev_tmpParticle->PSE, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.isStatic + index, &dev_tmpParticle->isStatic, sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.vol + index, &dev_tmpParticle->vol, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.circulation + index, &dev_tmpParticle->circulation, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);*/

    cudaFree(dev_tmpParticle);
}

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::removeParticleDevice(unsigned int index) {
    // not the last particle
    if (index != numParticles - 1) {
        cudaMemcpy(dev_particles.X + index, dev_particles.X + numParticles - 1, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.U + index, dev_particles.U + numParticles - 1, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.J + index, dev_particles.J + numParticles - 1, sizeof(vpmmat3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.Gamma + index, dev_particles.Gamma + numParticles - 1, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.sigma + index, dev_particles.sigma + numParticles - 1, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.SFS + index, dev_particles.SFS + numParticles - 1, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.C + index, dev_particles.C + numParticles - 1, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.M + index, dev_particles.M + numParticles - 1, sizeof(vpmmat3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.index + index, dev_particles.index + numParticles - 1, sizeof(int), cudaMemcpyDeviceToDevice);
        /*cudaMemcpy(dev_particles.PSE + index, dev_particles.PSE + numParticles, sizeof(vpmvec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.isStatic + index, dev_particles.isStatic + numParticles, sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.vol + index, dev_particles.vol + numParticles, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.circulation + index, dev_particles.circulation + numParticles, sizeof(vpmfloat), cudaMemcpyDeviceToDevice);*/

        synchronized = 0;
    }

    --numParticles;
}

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::syncParticlesDeviceToHost(int bufferMask, cudaStream_t stream) {
    _cpyParticleBuffer(particles, dev_particles, 0, numParticles, bufferMask & (~synchronized), stream);
    synchronized |= bufferMask;
}

template <typename R, typename S, typename K>
void ParticleField<R, S, K>::syncParticlesHostToDevice(int bufferMask, cudaStream_t stream) {
    _cpyParticleBuffer(dev_particles, particles, 0, numParticles, bufferMask & (~synchronized), stream);
    synchronized |= bufferMask;
} 

template <typename R, typename S, typename K>
ParticleField<R, S, K>::ParticleField(
    unsigned int maxParticles,
    ParticleBuffer particles,
    unsigned int numParticles,
    unsigned int timeStep,
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
    synchronized(0) {

    dev_particles.mallocFields(maxParticles, BUFFER_ALL);

    // Minimum requirement for initialization
    if (!(particles.bufferFields & (BUFFER_X | BUFFER_GAMMA | BUFFER_SIGMA))) {
        std::cerr << "Initialization particleBuffer does not have minimum required fields" << std::endl;
        exit(1);
    }
	syncParticlesHostToDevice(particles.bufferFields);
};

template <typename R, typename S, typename K>
ParticleField<R, S, K>::~ParticleField() {
    // free device memory
    dev_particles.freeFields(BUFFER_ALL);
}

// *************************************************************
// *                      RELAXATION                           *
// *************************************************************

template <typename R, typename S, typename K>
void PedrizzettiRelaxation::operator()(int N, ParticleField<R, S, K>& field, int numBlocks, int blockSize, cudaStream_t stream) {
    calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (PedrizzettiRelaxation) failed!");

    pedrizzettiRelax<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles, relaxFactor);
    checkCUDAError("PedrizzettiRelaxation failed!");
}

__global__ void pedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3 omega    = nablaCrossX(particles.J[index]);
    const vpmvec3 oldGamma = particles.Gamma[index];

    particles.Gamma[index] = (1.0f - relaxFactor) * oldGamma
        + relaxFactor * glm::length(oldGamma) / glm::length(omega) * omega;
}

template <typename R, typename S, typename K>
void CorrectedPedrizzettiRelaxation::operator()(int N, ParticleField<R, S, K>& field, int numBlocks, int blockSize, cudaStream_t stream) {
    calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (CorrectedPedrizzettiRelaxation) failed!");

    correctedPedrizzettiRelax<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles, relaxFactor);
    checkCUDAError("CorrectedPedrizzettiRelaxation failed!");
}

__global__ void correctedPedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3 omega      = nablaCrossX(particles.J[index]);
    const vpmvec3 oldGamma   = particles.Gamma[index];
    const vpmfloat omegaNorm = glm::length(omega);
    const vpmfloat gammaNorm = glm::length(oldGamma);

    const vpmfloat tmp = sqrt(1.0f - 2.0f * (1.0f - relaxFactor) * relaxFactor
        * (1.0f - glm::dot(oldGamma, omega) / (omegaNorm * gammaNorm)));

    particles.Gamma[index] = ((1.0f - relaxFactor) * oldGamma
        + relaxFactor * gammaNorm / omegaNorm * omega) / tmp;
}

// *************************************************************
// *                     SFS modeling                          *
// *************************************************************

__global__ void calculateTemporary(int N, ParticleBuffer particles, bool testFilter) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    if (testFilter) {
        particles.M[index][0] = xDotNablaY(particles.Gamma[index], particles.J[index]);
        particles.M[index][1] = particles.SFS[index];
    }
    else {
        particles.M[index][0] -= xDotNablaY(particles.Gamma[index], particles.J[index]);
        particles.M[index][1] -= particles.SFS[index];
    }
}

__global__ void calculateCoefficient(int N, ParticleBuffer particles, vpmfloat zeta0,
    vpmfloat alpha, vpmfloat relaxFactor, bool forcePositive, vpmfloat minC, vpmfloat maxC) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3 particleGamma  = particles.Gamma[index];
    const vpmvec3 particleSFS    = particles.SFS[index];
    const vpmmat3 particleM      = particles.M[index];
    const vpmfloat particleSigma = particles.sigma[index];

    vpmvec3 particleC = particles.C[index];

    vpmfloat numerator = glm::dot(particleM[0], particleGamma);
    numerator *= 3.0f * alpha - 2.0f;

    vpmfloat denominator = glm::dot(particleM[1], particleGamma);
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
    particles.C[index] = particleC;
}

template <typename R, typename S, typename K>
void DynamicSFS::operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize, cudaStream_t stream) {
    K& kernel = field.kernel;
    ParticleBuffer& particles = field.dev_particles;
    const int N = field.numParticles;

    if (a == 1.0f || a == 0.0f) {
        // CALCULATIONS WITH TEST FILTER
        calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, particles, particles, kernel, true, alpha);
        checkCUDAError("calcVelJacNaive (DynamicsSFS: test filter) failed!");

        calcEstrNaive<<<numBlocks, blockSize, 16 * blockSize * sizeof(vpmfloat), stream>>>(N, N, particles, particles, kernel, true, alpha);
        checkCUDAError("calcEstrNaive (DynamicsSFS: test filter) failed!");

        calculateTemporary<<<numBlocks, blockSize, 0, stream>>>(N, particles, true);
        checkCUDAError("calculateTemporary (DynamicsSFS: test filter) failed!");

        // CALCULATIONS WITH DOMAIN FILTER
        calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, particles, particles, kernel, true);
        checkCUDAError("calcVelJacNaive (DynamicsSFS: domain filter) failed!");

        calcEstrNaive<<<numBlocks, blockSize, 16 * blockSize * sizeof(vpmfloat), stream>>>(N, N, particles, particles, kernel, true);
        checkCUDAError("calcEstrNaive (DynamicsSFS: domain filter) failed!");

        calculateTemporary<<<numBlocks, blockSize, 0, stream>>>(N, particles, false);
        checkCUDAError("calculateTemporary (DynamicsSFS: domain filter) failed!");

        // CALCULATE COEFFICIENT
        calculateCoefficient<<<numBlocks, blockSize, 0, stream>>>(N, particles, kernel.zeta(0.0), alpha,
            relaxFactor, forcePositive, minC, maxC);
        checkCUDAError("calculateCoefficient failed!");
    }
    else {
        calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, particles, particles, kernel, true);
        checkCUDAError("calcVelJacNaive (DynamicsSFS: 2nd step) failed!");

        calcEstrNaive<<<numBlocks, blockSize, 16 * blockSize * sizeof(vpmfloat), stream>>>(N, N, particles, particles, kernel, true);
        checkCUDAError("calcEstrNaive (DynamicsSFS: 2nd step) failed!");
    }
}

template <typename R, typename S, typename K>
void NoSFS::operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize, cudaStream_t stream) {
    const int N = field.numParticles;

    cudaMemset(field.dev_particles.SFS, 0, N * sizeof(vpmvec3));
    checkCUDAError("cudaMemset (SFS reset) failed!");

    calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (NoSFS) failed!");
}

template <typename K>
__global__ void calcEstrNaive(int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset, vpmfloat testFilterFactor) {

    const int index = threadIdx.x + (blockIdx.x * blockDim.x);

    const int s_index = threadIdx.x;
    // number of vpmfloats per particle: 3 + 9 + 3 + 1 = 16
    extern __shared__ vpmfloat sharedMemory[];
    vpmvec3* s_sourceX = (vpmvec3*)sharedMemory;
    vpmmat3* s_sourceJ = (vpmmat3*)(s_sourceX + blockDim.x);
    vpmvec3* s_sourceGammaSigma = (vpmvec3*)(s_sourceJ + blockDim.x);
    vpmfloat* s_sourceInvSigma = (vpmfloat*)(s_sourceGammaSigma + blockDim.x);

    // Get required variables from global memory
    vpmvec3 targetX;
    vpmmat3 targetJ;
    vpmvec3 targetSFS;
    if (index < targetN) {
        targetX = targetParticles.X[index];
        targetJ = targetParticles.J[index];
        if (reset) {
            targetSFS = vpmvec3{ 0.0f };
        }
        else {
            targetSFS = targetParticles.SFS[index];
        }
    }
    else {
        targetX = vpmvec3{ 0.0f };
        targetJ = vpmmat3{ 0.0f };
        targetSFS = vpmvec3{ 0.0f };
    }

    vpmvec3 targetXSigma{ 0.0f };
    for (int j = 0; j < sourceN; j += blockDim.x) {
        if (j + s_index < sourceN) {
            s_sourceInvSigma[s_index] = 1.0f / (sourceParticles.sigma[s_index + j] * testFilterFactor);
            s_sourceX[s_index] = sourceParticles.X[s_index + j] * s_sourceInvSigma[s_index];
            s_sourceJ[s_index] = sourceParticles.J[s_index + j];
            s_sourceGammaSigma[s_index] = sourceParticles.Gamma[s_index + j]
                * s_sourceInvSigma[s_index] * s_sourceInvSigma[s_index] * s_sourceInvSigma[s_index];
            targetXSigma = targetX * s_sourceInvSigma[s_index];
        }
        __syncthreads();

        for (int i = 0; (i < blockDim.x) && (j + i < sourceN); ++i) {
            targetSFS += kernel.zeta(glm::length(targetXSigma - s_sourceX[i]))
                * xDotNablaY(s_sourceGammaSigma[i], targetJ - s_sourceJ[i]);
        }

        __syncthreads();
    }

    // Copy variables back to global memory
    if (index < targetN) {
        targetParticles.SFS[index] = targetSFS;
    }
}

template <typename K>
__global__ void calcVelJacNaive(int targetN, int sourceN, ParticleBuffer targetParticles, 
    ParticleBuffer sourceParticles, K kernel, bool reset, vpmfloat testFilterFactor) {

    const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	const vpmfloat invTestFilterFactor = 1.0f / testFilterFactor;

    const int s_index = threadIdx.x;
    extern __shared__ vpmfloat sharedMemory[];
    vpmvec3*  s_sourceX     = (vpmvec3*)sharedMemory;
    vpmvec3*  s_sourceGamma = (vpmvec3*)(s_sourceX + blockDim.x);
    vpmfloat* s_sourceInvSigma = (vpmfloat*)(s_sourceGamma + blockDim.x);

    vpmvec3 targetX;
    vpmvec3 targetU;
    vpmmat3 targetJ;
    if (index < targetN) {
        // Get target variables from global memory
        targetX = targetParticles.X[index];

        if (reset) {
            targetU = vpmvec3{ 0.0f };
            targetJ = vpmmat3{ 0.0f };
        }
        else {
            targetU = targetParticles.U[index];
            targetJ = targetParticles.J[index];
        }
    }
    else {
		targetX = vpmvec3{ 0.0f };
		targetU = vpmvec3{ 0.0f };
		targetJ = vpmmat3{ 0.0f };
    }

    // Copy source variables into shared memory
    for (int j = 0; j < sourceN; j += blockDim.x) {
        if (j + s_index < sourceN) {
            s_sourceX[s_index]     = sourceParticles.X[s_index + j];
            s_sourceGamma[s_index] = sourceParticles.Gamma[s_index + j];
            s_sourceInvSigma[s_index] = invTestFilterFactor / sourceParticles.sigma[s_index + j];
        }
        __syncthreads();

        for (int i = 0; (i < blockDim.x) && (j + i < sourceN); ++i) {
            vpmvec3 dX = targetX - s_sourceX[i];
            vpmfloat r = glm::length(dX);
            const vpmfloat invSourceSigma = r * s_sourceInvSigma[i];
			vpmvec3 sourceGamma = s_sourceGamma[i];
            
            if (r == 0.0f) continue;
            const vpmfloat invR = 1.0f / r;

            // Kernel evaluation
			const vpmvec2 g_dgdr = kernel.g_dgdr(invSourceSigma);

            const vpmfloat tmp = -const4 * (invR * invR * invR);

            // Compute velocity
            const vpmvec3 crossProd = tmp * glm::cross(dX, sourceGamma);
            targetU += g_dgdr[0] * crossProd;

            // Compute Jacobian
            dX *= (g_dgdr[1] * invSourceSigma - 3.0f * g_dgdr[0]) * (invR * invR);

            targetJ += glm::outerProduct(crossProd, dX);
            sourceGamma *= tmp * g_dgdr[0];

            // Account for kronecker delta term
            targetJ[0][1] -= sourceGamma[2];
            targetJ[0][2] += sourceGamma[1];
            targetJ[1][0] += sourceGamma[2];
            targetJ[1][2] -= sourceGamma[0];
            targetJ[2][0] -= sourceGamma[1];
            targetJ[2][1] += sourceGamma[0];
        }

        __syncthreads();
    }

    if (index < targetN) {
        // Copy variables back to global memory
        targetParticles.U[index] = targetU;
        targetParticles.J[index] = targetJ;
    }
}

__global__ void rungeKuttaStep(int N, ParticleBuffer particles, vpmfloat a, vpmfloat b, vpmfloat dt, vpmfloat zeta0, vpmvec3 Uinf) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmfloat particleC   = particles.C[index][0];
    const vpmvec3  particleU   = particles.U[index];
    const vpmvec3  particleSFS = particles.SFS[index];
    const vpmmat3  particleJ   = particles.J[index];
    
    vpmfloat particleSigma = particles.sigma[index];
    vpmvec3  particleGamma = particles.Gamma[index];
    vpmvec3  particleX     = particles.X[index];
    vpmmat3  particleM;
    if (a == 1.0f || a == 0.0f) {
        particleM = vpmmat3{ 0.0f };
    }
    else {
        particleM = particles.M[index];
    }

    // Position update
    particleM[0] = a * particleM[0] + dt * (particleU + Uinf);
    particleX += b * particleM[0];
    particles.X[index] = particleX;

    vpmvec3 S = xDotNablaY(particleGamma, particleJ);
#ifdef CLASSIC_VPM
    vpmfloat Z = 0.0f;
#else
    vpmfloat Z = (vpmfloat)0.2 * glm::dot(S, particleGamma) / glm::dot(particleGamma, particleGamma);
#endif

    // Gamma update
    particleM[1] = a * particleM[1] + dt * (S - 3.0f * Z * particleGamma
        - particleC * particleSFS * particleSigma * particleSigma * particleSigma / zeta0);
    particleGamma += b * particleM[1];
    particles.Gamma[index] = particleGamma;

#ifndef CLASSIC_VPM
    // Sigma update
    particleM[2][1] = a * particleM[2][1] - dt * (particleSigma * Z);
    particleSigma += b * particleM[2][1];

    particles.sigma[index] = particleSigma;
#endif

    particles.M[index] = particleM;
}

template <typename R, typename S, typename K>
void rungeKutta(ParticleField<R, S, K>& field, vpmfloat dt, bool useRelax, int numBlocks, int blockSize, cudaStream_t stream) {

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
        field.sfs(field, a, b, numBlocks, blockSize, stream);

        rungeKuttaStep<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles, a, b, dt, kernel.zeta(0.0), field.uInf);
        checkCUDAError("rungeKuttaStep failed!");
    }

    field.relaxation(N, field, numBlocks, blockSize, stream);

    ++field.timeStep;
    field.synchronized = false;
}

int outputMaskToBufferMask(int outputMask) {
    int bufferMask = 0;
    if (outputMask & OUTPUT_X) bufferMask |= BUFFER_X;
    if (outputMask & OUTPUT_U) bufferMask |= BUFFER_U;
    if (outputMask & OUTPUT_OMEGA) bufferMask |= BUFFER_J;
    if (outputMask & OUTPUT_SIGMA) bufferMask |= BUFFER_SIGMA;
    if (outputMask & OUTPUT_GAMMA) bufferMask |= BUFFER_GAMMA;
    if (outputMask & OUTPUT_INDEX) bufferMask |= BUFFER_INDEX;

    return bufferMask;
}

//template <typename R, typename S, typename K>
//void writeVTK(ParticleField<R, S, K>& field, const std::string& filename, int outputMask) {
//    const int dim = 3;
//
//    static leanvtk::VTUWriter writer;
//
//    static std::vector<double> particleX;
//    static std::vector<double> particleU;
//    static std::vector<double> particleGamma;
//    static std::vector<double> particleOmega;
//    static std::vector<double> particleSigma;
//    static std::vector<double> particleIdx;
//
//    if (outputMask & OUTPUT_X) {
//        particleX.insert(
//            particleX.end(),
//            (vpmfloat*)field.particles.X,
//            (vpmfloat*)(field.particles.X + field.numParticles)
//        );
//
//        writer.add_vector_field("position", particleX, dim);
//    }
//    if (outputMask & OUTPUT_U) {
//        particleU.insert(
//            particleU.end(),
//            (vpmfloat*)field.particles.U,
//            (vpmfloat*)(field.particles.U + field.numParticles)
//        );
//
//        writer.add_vector_field("velocity", particleU, dim);
//    }
//    if (outputMask & OUTPUT_GAMMA) {
//        particleGamma.insert(
//            particleGamma.end(),
//            (vpmfloat*)field.particles.Gamma,
//            (vpmfloat*)(field.particles.Gamma + field.numParticles)
//        );
//
//        writer.add_vector_field("circulation", particleGamma, dim);
//    }
//    if (outputMask & OUTPUT_SIGMA) {
//        particleSigma.insert(
//            particleSigma.end(),
//            field.particles.sigma,
//            field.particles.sigma + field.numParticles
//        );
//
//        writer.add_scalar_field("sigma", particleSigma);
//    }
//    if (outputMask & OUTPUT_INDEX) {
//        particleIdx.insert(
//            particleIdx.end(),
//            field.particles.index,
//            field.particles.index + field.numParticles
//        );
//
//        writer.add_scalar_field("index", particleIdx);
//    }
//    if (outputMask & OUTPUT_OMEGA) {
//        particleOmega.reserve(field.maxParticles * dim);
//
//        vpmvec3 omega;
//        for (int i = 0; i < field.numParticles; ++i) {
//            omega = nablaCrossX(field.particles.J[i]);
//            particleOmega.insert(particleOmega.end(), (vpmfloat*)&omega, (vpmfloat*)&omega + 3);
//        }
//
//        writer.add_vector_field("vorticity", particleOmega, dim);
//    }
// 
//    writer.write_point_cloud("../output/" + filename + "_" + std::to_string(field.timeStep) + ".vtu", dim, particleX);
//    writer.clear();
//
//    particleX.clear();
//    particleU.clear();
//    particleGamma.clear();
//    particleSigma.clear();
//    particleIdx.clear();
//    particleOmega.clear();
//}

void writeVTK(ParticleBuffer &particles, int N, const std::string& filename, int outputMask) {
    const int dim = 3;

    static leanvtk::VTUWriter writer;

    static std::vector<double> particleX;
    static std::vector<double> particleU;
    static std::vector<double> particleGamma;
    static std::vector<double> particleOmega;
    static std::vector<double> particleSigma;
    static std::vector<double> particleIdx;

    if (outputMask & OUTPUT_X) {
        particleX.insert(
            particleX.end(),
            (vpmfloat*)particles.X,
            (vpmfloat*)(particles.X + N)
        );

        writer.add_vector_field("position", particleX, dim);
    }
    if (outputMask & OUTPUT_U) {
        particleU.insert(
            particleU.end(),
            (vpmfloat*)particles.U,
            (vpmfloat*)(particles.U + N)
        );

        writer.add_vector_field("velocity", particleU, dim);
    }
    if (outputMask & OUTPUT_GAMMA) {
        particleGamma.insert(
            particleGamma.end(),
            (vpmfloat*)particles.Gamma,
            (vpmfloat*)(particles.Gamma + N)
        );

        writer.add_vector_field("circulation", particleGamma, dim);
    }
    if (outputMask & OUTPUT_SIGMA) {
        particleSigma.insert(
            particleSigma.end(),
            particles.sigma,
            particles.sigma + N
        );

        writer.add_scalar_field("sigma", particleSigma);
    }
    if (outputMask & OUTPUT_INDEX) {
        particleIdx.insert(
            particleIdx.end(),
            particles.index,
            particles.index + N
        );

        writer.add_scalar_field("index", particleIdx);
    }
    if (outputMask & OUTPUT_OMEGA) {
        particleOmega.reserve(N * dim);

        vpmvec3 omega;
        for (int i = 0; i < N; ++i) {
            omega = nablaCrossX(particles.J[i]);
            particleOmega.insert(particleOmega.end(), (vpmfloat*)&omega, (vpmfloat*)&omega + 3);
        }

        writer.add_vector_field("vorticity", particleOmega, dim);
    }

    writer.write_point_cloud("../output/" + filename + ".vtu", dim, particleX);
    writer.clear();

    particleX.clear();
    particleU.clear();
    particleGamma.clear();
    particleSigma.clear();
    particleIdx.clear();
    particleOmega.clear();
}

template <typename R, typename S, typename K>
void calcVortexRingMetrics(ParticleField<R, S, K>& field, int iteration, std::string filename, int numRings = 2) {
    field.syncParticlesDeviceToHost(BUFFER_X | BUFFER_GAMMA);
    int numParticlesRing = field.numParticles / numRings;

    std::vector<vpmfloat> ringRadii;
    std::vector<vpmvec3>  ringCenters;

    for (int j = 0; j < numRings; ++j) {
        int offset = j * numParticlesRing;
        // Calculate ring center
        vpmvec3 ringCenter = vpmvec3{ 0 };
        vpmfloat totalGamma = 0;
        for (int i = offset; i < numParticlesRing + offset; ++i) {
            vpmfloat Gamma = glm::length(field.particles.Gamma[i]);
            totalGamma += Gamma;
            ringCenter += Gamma * field.particles.X[i];
        }
        ringCenter /= totalGamma;

        // Calculate ring radius
        vpmfloat ringRadius = 0;
        for (int i = offset; i < numParticlesRing + offset; ++i) {
            vpmfloat Gamma = glm::length(field.particles.Gamma[i]);
            vpmfloat radius = glm::length(field.particles.X[i] - ringCenter);
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
    unsigned int maxParticles,
    unsigned int numParticles,
    unsigned int numTimeSteps,
    vpmfloat dt,
    unsigned int fileSaveSteps,
    vpmvec3 uInf,
    ParticleBuffer particleBuffer,
    R relaxation,
    S sfs,
    K kernel,
    int blockSize,
    std::string filename) {

    int numBlocks = (numParticles + blockSize - 1) / blockSize;

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

    int outputMask = OUTPUT_ALL;
    int bufferMask = outputMaskToBufferMask(outputMask);

    ParticleBuffer outputBufferHost{ BUFFER_HOST_PINNED };
    outputBufferHost.mallocFields(maxParticles, bufferMask);

    for (int i = 0; i < numTimeSteps + 1; ++i) {
        // calcVortexRingMetrics(field, i, "test");

        rungeKutta(field, dt, true, numBlocks, blockSize);

        if ((fileSaveSteps != 0) && (i % fileSaveSteps == 0)) {
            writeVTK(outputBufferHost, field.numParticles, filename + "_" + std::to_string(field.timeStep), outputMask);

            std::cout << outputBufferHost.U[0].x << std::endl;

            _cpyParticleBuffer(outputBufferHost, field.dev_particles, 0, field.numParticles, bufferMask);

            cudaDeviceSynchronize();
        }
    }
}

template <typename R, typename S, typename K>
void runBoundaryVPM(
    unsigned int maxParticles,
    unsigned int numParticles,
    unsigned int numBoundary,
    unsigned int numTimeSteps,
    vpmfloat dt,
    unsigned int fileSaveSteps,
    vpmvec3 uInf,
    ParticleBuffer particleBuffer,
    const ParticleBuffer boundaryBuffer,
    R relaxation,
    S sfs,
    K kernel,
    int blockSize,
    std::string filename) {

    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    int outputMask = OUTPUT_ALL;
    int bufferMask = outputMaskToBufferMask(outputMask);
    int boundaryMask = BUFFER_X | BUFFER_SIGMA | BUFFER_GAMMA | BUFFER_INDEX;

    ParticleBuffer dev_boundaryBuffer{ BUFFER_DEVICE };
    dev_boundaryBuffer.mallocFields(numBoundary, boundaryMask);

    // Copy boundary particle buffer from host to device
    cpyParticleBuffer(dev_boundaryBuffer, boundaryBuffer, numBoundary, numBoundary, numBoundary, 0, boundaryMask);

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

    unsigned int boundaryIndex = numParticles;

    for (int i = 0; i < numTimeSteps; ++i) {
        std::cout << field.numParticles << " " << boundaryIndex << " " << field.particles.U[0].x << std::endl;

        rungeKutta(field, dt, true, numBlocks, blockSize);

        if ((fileSaveSteps != 0) && (i % fileSaveSteps == 0)) {
            // writeVTK(field, filename, outputMask);

            std::cout << field.particles.U[0].x << std::endl;

            field.syncParticlesDeviceToHost(bufferMask);
        }

        if (boundaryIndex + numBoundary >= field.maxParticles) boundaryIndex = 0;

        field.cpyParticlesDeviceToDevice(dev_boundaryBuffer, numBoundary, boundaryIndex, boundaryMask);

        numBlocks = (field.numParticles + blockSize - 1) / blockSize;

        boundaryIndex += numBoundary;
    }
}

void runSimulation() {
    // Define basic parameters
    unsigned int maxParticles = 50000;
    unsigned int numTimeSteps = 50;
    vpmfloat dt = 1e-2;
    unsigned int numStepsVTK = 1;
    vpmvec3 uInf{ 0, 0, 0 };
    int blockSize = 64;
    const int simulationType = 0;

    // Create host particle buffer
#ifdef PINNED_MEMORY
    ParticleBuffer particleBuffer{ BUFFER_HOST_PINNED };
#else
    ParticleBuffer particleBuffer{ BUFFER_HOST };
#endif
    int inputBufferMask = BUFFER_X | BUFFER_U | BUFFER_J | BUFFER_INDEX | BUFFER_GAMMA | BUFFER_SIGMA;
    particleBuffer.mallocFields(maxParticles, inputBufferMask);

    int numParticles;
    switch (simulationType)
    {
    case 0:
        numParticles = initVortexRings(particleBuffer, maxParticles);
        break;

    case 1: {
        // Create host boundary buffer
        ParticleBuffer boundaryBuffer{ BUFFER_HOST_PINNED };
        boundaryBuffer.mallocFields(maxParticles, BUFFER_X | BUFFER_GAMMA | BUFFER_SIGMA | BUFFER_INDEX);

        std::pair<unsigned int, unsigned int> numbers = initRoundJet(particleBuffer, boundaryBuffer, maxParticles);
        numParticles = numbers.first;
        unsigned int numBoundary = numbers.second;

        // Run VPM method
        runBoundaryVPM(
            maxParticles,
            numParticles,
            numBoundary,
            numTimeSteps,
            dt,
            numStepsVTK,
            uInf,
            particleBuffer,
            boundaryBuffer,
            PedrizzettiRelaxation(0.3),
            DynamicSFS(),
            GaussianErfKernel(),
            blockSize,
            "test"
        );

        boundaryBuffer.freeFields(boundaryBuffer.bufferFields);
        particleBuffer.freeFields(particleBuffer.bufferFields);
        return;
    }
    }

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
        DynamicSFS(),
        WinckelmansKernel(),
        blockSize,
        "test"
    );

    particleBuffer.freeFields(particleBuffer.bufferFields);
}