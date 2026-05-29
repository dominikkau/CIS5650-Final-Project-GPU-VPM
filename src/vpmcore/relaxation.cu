#include "relaxation.h"
#include "vpmmain.h"

void PedrizzettiRelaxation::operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream) {
    const vpm::pidx_t N = field.numParticles;
    const CUDAKernelParams params{ numBlocks, blockSize, 7 * blockSize * sizeof(vpm::real), stream };
    calcVelJacNaiveWrapper(params, N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (PedrizzettiRelaxation) failed!");

    pedrizzettiRelax<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles.J(), field.dev_particles.Gamma(), relaxFactor);
    checkCUDAError("PedrizzettiRelaxation failed!");
}

__global__ void pedrizzettiRelax(vpm::pidx_t N, const vpm::mat3* __restrict__ J, vpm::vec3* __restrict__ Gamma, vpm::real relaxFactor) {
    vpm::pidx_t index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpm::vec3 omega    = nablaCrossX(J[index]);
    const vpm::vec3 oldGamma = Gamma[index];

    Gamma[index] = (1.0f - relaxFactor) * oldGamma
        + relaxFactor * glm::length(oldGamma) / glm::length(omega) * omega;
}

void CorrectedPedrizzettiRelaxation::operator()(ParticleField& field, int numBlocks, int blockSize, cudaStream_t stream) {
    const vpm::pidx_t N = field.numParticles;
    const CUDAKernelParams params{ numBlocks, blockSize, 7 * blockSize * sizeof(vpm::real), stream };
    calcVelJacNaiveWrapper(params, N, N, field.dev_particles, field.dev_particles, field.kernel, true);
    checkCUDAError("calcVelJacNaive (CorrectedPedrizzettiRelaxation) failed!");

    correctedPedrizzettiRelax<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles.J(), field.dev_particles.Gamma(), relaxFactor);
    checkCUDAError("CorrectedPedrizzettiRelaxation failed!");
}

__global__ void correctedPedrizzettiRelax(vpm::pidx_t N, const vpm::mat3* __restrict__ J, vpm::vec3* __restrict__ Gamma, vpm::real relaxFactor) {
    vpm::pidx_t index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpm::vec3 omega      = nablaCrossX(J[index]);
    const vpm::vec3 oldGamma   = Gamma[index];
    const vpm::real omegaNorm = glm::length(omega);
    const vpm::real gammaNorm = glm::length(oldGamma);

    const vpm::real tmp = sqrt(1.0f - 2.0f * (1.0f - relaxFactor) * relaxFactor
        * (1.0f - glm::dot(oldGamma, omega) / (omegaNorm * gammaNorm)));

    Gamma[index] = ((1.0f - relaxFactor) * oldGamma
        + relaxFactor * gammaNorm / omegaNorm * omega) / tmp;
}