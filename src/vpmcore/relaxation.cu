#include "relaxation.h"
#include "kernel.h"

void PedrizzettiRelaxation::operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream) {
    calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (PedrizzettiRelaxation) failed!");

    this->relax<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles, relaxFactor);
    checkCUDAError("PedrizzettiRelaxation failed!");
}

static __global__ void PedrizzettiRelaxation::relax(int N, ParticleBuffer particles, vpmfloat relaxFactor) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpmvec3 omega    = nablaCrossX(particles.J[index]);
    const vpmvec3 oldGamma = particles.Gamma[index];

    particles.Gamma[index] = (1.0f - relaxFactor) * oldGamma
        + relaxFactor * glm::length(oldGamma) / glm::length(omega) * omega;
}

void CorrectedPedrizzettiRelaxation::operator()(int N, ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream) {
    calcVelJacNaive<<<numBlocks, blockSize, 7 * blockSize * sizeof(vpmfloat), stream>>>(N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (CorrectedPedrizzettiRelaxation) failed!");

    this->relax<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles, relaxFactor);
    checkCUDAError("CorrectedPedrizzettiRelaxation failed!");
}

static __global__ void CorrectedPedrizzettiRelaxation::relax(int N, ParticleBuffer particles, vpmfloat relaxFactor) {
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