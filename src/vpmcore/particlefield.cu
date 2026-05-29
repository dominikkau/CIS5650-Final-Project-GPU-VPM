#include <iostream>
#include "particlefield.h"

__host__ __device__ void Particle::reset() {
    U = vpm::vec3{ 0.0f };
    J = vpm::mat3{ 0.0f };
    //PSE = vpm::vec3{ 0.0f };
}

__host__ __device__ void Particle::resetSFS() {
    SFS = vpm::vec3{ 0.0f };
}

void ParticleField::cpyParticlesDeviceToDevice(ParticleBuffer srcBuffer, vpm::pidx_t srcIndex, vpm::pidx_t count,
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

	cudaMemcpy(dev_particles.X() + numParticles, &dev_tmpParticle->X, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.U() + numParticles, &dev_tmpParticle->U, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.J() + numParticles, &dev_tmpParticle->J, sizeof(vpm::mat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.Gamma() + numParticles, &dev_tmpParticle->Gamma, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.sigma() + numParticles, &dev_tmpParticle->sigma, sizeof(vpm::real), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.SFS() + numParticles, &dev_tmpParticle->SFS, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.C() + numParticles, &dev_tmpParticle->C, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.M() + numParticles, &dev_tmpParticle->M, sizeof(vpm::mat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_particles.index() + numParticles, &dev_tmpParticle->index, sizeof(vpm::pidx_t), cudaMemcpyDeviceToDevice);
    /*cudaMemcpy(dev_particles.PSE() + numParticles, &dev_tmpParticle->PSE, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.isStatic() + numParticles, &dev_tmpParticle->isStatic, sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.vol() + numParticles, &dev_tmpParticle->vol, sizeof(vpm::real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.circulation() + numParticles, &dev_tmpParticle->circulation, sizeof(vpm::real), cudaMemcpyDeviceToDevice);*/

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

    cudaMemcpy(dev_particles.X() + index, &dev_tmpParticle->X, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.U() + index, &dev_tmpParticle->U, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.J() + index, &dev_tmpParticle->J, sizeof(vpm::mat3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.Gamma() + index, &dev_tmpParticle->Gamma, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.sigma() + index, &dev_tmpParticle->sigma, sizeof(vpm::real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.SFS() + index, &dev_tmpParticle->SFS, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.C() + index, &dev_tmpParticle->C, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.M() + index, &dev_tmpParticle->M, sizeof(vpm::mat3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.index() + index, &dev_tmpParticle->index, sizeof(vpm::pidx_t), cudaMemcpyDeviceToDevice);
    /*cudaMemcpy(dev_particles.PSE() + index, &dev_tmpParticle->PSE, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.isStatic() + index, &dev_tmpParticle->isStatic, sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.vol() + index, &dev_tmpParticle->vol, sizeof(vpm::real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_particles.circulation() + index, &dev_tmpParticle->circulation, sizeof(vpm::real), cudaMemcpyDeviceToDevice);*/

    cudaFree(dev_tmpParticle);
}

void ParticleField::removeParticleDevice(vpm::pidx_t index) {
    // not the last particle
    if (index != numParticles - 1) {
        cudaMemcpy(dev_particles.X() + index, dev_particles.X() + numParticles - 1, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.U() + index, dev_particles.U() + numParticles - 1, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.J() + index, dev_particles.J() + numParticles - 1, sizeof(vpm::mat3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.Gamma() + index, dev_particles.Gamma() + numParticles - 1, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.sigma() + index, dev_particles.sigma() + numParticles - 1, sizeof(vpm::real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.SFS() + index, dev_particles.SFS() + numParticles - 1, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.C() + index, dev_particles.C() + numParticles - 1, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.M() + index, dev_particles.M() + numParticles - 1, sizeof(vpm::mat3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.index() + index, dev_particles.index() + numParticles - 1, sizeof(vpm::pidx_t), cudaMemcpyDeviceToDevice);
        /*cudaMemcpy(dev_particles.PSE() + index, dev_particles.PSE() + numParticles, sizeof(vpm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.isStatic() + index, dev_particles.isStatic() + numParticles, sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.vol() + index, dev_particles.vol() + numParticles, sizeof(vpm::real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_particles.circulation() + index, dev_particles.circulation() + numParticles, sizeof(vpm::real), cudaMemcpyDeviceToDevice);*/

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
    ParticleBuffer particles,
    vpm::pidx_t numParticles,
    unsigned int timeStep,
    KernelType kernel,
    vpm::vec3 uInf,
    std::unique_ptr<SFSScheme> sfs,
    std::unique_ptr<RelaxationScheme> relaxation)
    :
    particles(particles),
    numParticles(numParticles),
    timeStep(timeStep),
    kernel(kernel),
    uInf(uInf),
    sfs(std::move(sfs)),
    relaxation(std::move(relaxation)),
    synchronized(0),
    dev_particles(ParticleBuffer(ParticleBufferType::DEVICE, particles.size()))
{
    // Minimum requirement for initialization
    if (!(particles.fields() & (BufferField::X | BufferField::GAMMA | BufferField::SIGMA))) {
        std::cerr << "Initialization particleBuffer does not have minimum required fields" << std::endl;
        exit(1);
    }

    dev_particles.mallocFields(BufferField::ALL);
	syncParticlesHostToDevice(particles.fields());
};

ParticleField::~ParticleField() {
    // free device memory
    dev_particles.freeFields(BufferField::ALL);
}