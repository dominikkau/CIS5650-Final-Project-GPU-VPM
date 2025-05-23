#include "particlefield.h"

void ParticleField::cpyParticlesDeviceToDevice(ParticleBuffer inParticles, unsigned int inNumParticles, 
    unsigned int startIndex, int bufferMask) {

    numParticles = cpyParticleBuffer(dev_particles, inParticles, numParticles,
        maxParticles, inNumParticles, startIndex, bufferMask);
}

void ParticleField::addParticleDevice(Particle& particle) {
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

void ParticleField::overwriteParticleDevice(Particle& particle, unsigned int index) {
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

void ParticleField::removeParticleDevice(unsigned int index) {
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

void ParticleField::syncParticlesDeviceToHost(int bufferMask, cudaStream_t stream) {
    _cpyParticleBuffer(particles, dev_particles, 0, numParticles, bufferMask & (~synchronized), stream);
    synchronized |= bufferMask;
}

void ParticleField::syncParticlesHostToDevice(int bufferMask, cudaStream_t stream) {
    _cpyParticleBuffer(dev_particles, particles, 0, numParticles, bufferMask & (~synchronized), stream);
    synchronized |= bufferMask;
}

ParticleField::ParticleField(
    unsigned int maxParticles,
    ParticleBuffer particles,
    unsigned int numParticles,
    unsigned int timeStep,
    std::unique_ptr<Kernel> kernel,
    vpmvec3 uInf,
    std::unique_ptr<SFSScheme> sfs,
    std::unique_ptr<RelaxationScheme> relaxation)
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

    dev_particles.mallocFields(maxParticles, BufferField::ALL);

    // Minimum requirement for initialization
    if (!(particles.bufferFields & (BufferField::X | BufferField::GAMMA | BufferField::SIGMA))) {
        std::cerr << "Initialization particleBuffer does not have minimum required fields" << std::endl;
        exit(1);
    }
	syncParticlesHostToDevice(particles.bufferFields);
};

ParticleField::~ParticleField() {
    // free device memory
    dev_particles.freeFields(BufferField::ALL);
}