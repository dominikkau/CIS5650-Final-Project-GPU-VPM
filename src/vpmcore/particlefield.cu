#include <iostream>
#include "particlefield.h"

__host__ __device__ void Particle::reset() {
    U_ = vpm::vec3{ 0.0f };
    J_ = vpm::mat3{ 0.0f };
    //PSE = vpm::vec3{ 0.0f };
}

__host__ __device__ void Particle::resetSFS() {
    SFS_ = vpm::vec3{ 0.0f };
}

void ParticleField::cpyParticlesDeviceToDevice(ParticleBuffer& srcBuffer, vpm::pidx_t srcIndex, vpm::pidx_t count,
    int bufferMask) {

    numParticles += cpyParticleBuffer(dev_particles, srcBuffer, numParticles,
        srcIndex, count, bufferMask);
}

void ParticleField::addParticleDevice(Particle& particle) {
    if (numParticles == dev_particles.size()) return;

    Particle* dev_tmpParticle;
	cudaMalloc((void**)&dev_tmpParticle, sizeof(Particle));
	checkCUDAError("cudaMalloc of dev_tmpParticle failed!");

	cudaMemcpy(dev_tmpParticle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy of dev_tmpParticle failed!");

    dev_particles.forEachFieldPair(*dev_tmpParticle,
        [this](auto* ptrA, auto* ptrB)
        {
            cudaMemcpy(ptrA + numParticles, ptrB, sizeof(*ptrA), cudaMemcpyDeviceToDevice);
        }, dev_particles.fields()
    );

    ++numParticles;

	cudaFree(dev_tmpParticle);
}

void ParticleField::overwriteParticleDevice(Particle& particle, vpm::pidx_t index) {
    if (index > numParticles) {
        addParticleDevice(particle);
        return;
    }

    Particle* dev_tmpParticle;
    cudaMalloc((void**)&dev_tmpParticle, sizeof(Particle));
    checkCUDAError("cudaMalloc of dev_tmpParticle failed!");

    cudaMemcpy(dev_tmpParticle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy of dev_tmpParticle failed!");

    dev_particles.forEachFieldPair(*dev_tmpParticle,
        [index, this](auto* ptrA, auto* ptrB)
        {
            cudaMemcpy(ptrA + index, ptrB, sizeof(*ptrA), cudaMemcpyDeviceToDevice);
        }, dev_particles.fields()
    );

    cudaFree(dev_tmpParticle);
}

void ParticleField::removeParticleDevice(vpm::pidx_t index) {
    // not the last particle
    if (index != numParticles - 1) {

        dev_particles.forEachFieldPair(dev_particles,
            [index, this](auto* ptrA, const auto* ptrB)
            {
                cudaMemcpy(ptrA + index, ptrB + numParticles - 1, sizeof(*ptrA), cudaMemcpyDeviceToDevice);
            }, dev_particles.fields()
        );

        synchronized = 0;
    }

    --numParticles;
}

void ParticleField::syncParticlesDeviceToHost(int bufferMask, cudaStream_t stream) {
    cpyParticleBuffer(particles, dev_particles, bufferMask & (~synchronized), stream);
    synchronized |= bufferMask;
}

void ParticleField::syncParticlesHostToDevice(int bufferMask, cudaStream_t stream) {
    cpyParticleBuffer(dev_particles, particles, bufferMask & (~synchronized), stream);
    synchronized |= bufferMask;
}

ParticleField::ParticleField(
    ParticleBuffer&& particles,
    vpm::pidx_t numParticles,
    unsigned int timeStep,
    KernelType kernel,
    vpm::vec3 uInf,
    std::unique_ptr<SFSScheme> sfs,
    std::unique_ptr<RelaxationScheme> relaxation)
    :
    particles(std::move(particles)),
    numParticles(numParticles),
    timeStep(timeStep),
    kernel(kernel),
    uInf(uInf),
    sfs(std::move(sfs)),
    relaxation(std::move(relaxation)),
    synchronized(0),
    dev_particles(ParticleBuffer( ParticleBufferType::DEVICE, particles.size() ))
{
    // Minimum requirement for initialization
    if (!(particles.fields() & (BufferField::X | BufferField::GAMMA | BufferField::SIGMA))) {
        std::cerr << "Initialization particleBuffer does not have minimum required fields" << std::endl;
        exit(1);
    }

    dev_particles.mallocFields(BufferField::ALL);
	syncParticlesHostToDevice(particles.fields());
};