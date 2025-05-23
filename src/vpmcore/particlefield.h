#pragma once

#include <memory>
#include "common.h"
#include "particlebuffer.h"
#include "kernels.h"
#include "sfs.h"
#include "relaxation.h"

enum class OutputType {
    NONE = 0,
    X = 1 << 0,
    U = 1 << 1,
    OMEGA = 1 << 2,
    GAMMA = 1 << 3,
    SIGMA = 1 << 4,
    INDEX = 1 << 5,
    ALL = 0xFFFF
};

// ParticleField definition
struct ParticleField {
    unsigned int maxParticles;           // Maximum number of particles
    ParticleBuffer particles;        // Pointer to host particle buffer
    ParticleBuffer dev_particles{ ParticleBufferType::DEVICE }; // Pointer to device particle buffer
    unsigned int numParticles;           // Number of particles in the field
    unsigned int timeStep;               // Current time step
    std::unique_ptr<Kernel> kernel;                   // Vortex particle kernel
    vpmvec3 uInf;               // Uniform freestream function
    std::unique_ptr<SFSScheme> sfs;                      // Subfilter-scale contributions scheme
    std::unique_ptr<RelaxationScheme> relaxation;               // Relaxation scheme
    int synchronized;           // Flags if host buffers are synchronized with device

    // Constructor
    ParticleField(
        unsigned int maxParticles,
        ParticleBuffer particles,
        unsigned int numParticles,
        unsigned int timeStep = 0,
        std::unique_ptr<Kernel> kernel = std::make_unique<GaussianErfKernel>(),
        vpmvec3 uInf = vpmvec3(0, 0, 0),
        std::unique_ptr<SFSScheme> sfs = std::make_unique<NoSFS>(),
        std::unique_ptr<RelaxationScheme> relaxation = std::make_unique<PedrizzettiRelaxation>(0.005f)
    );
    // Destructor
    ~ParticleField();

	void syncParticlesDeviceToHost(int bufferMask, cudaStream_t stream = 0);
	void syncParticlesHostToDevice(int bufferMask, cudaStream_t stream = 0);

    void addParticleDevice(Particle& particle);
    void overwriteParticleDevice(Particle& particle, unsigned int index);
    void removeParticleDevice(unsigned int index);
    void cpyParticlesDeviceToDevice(ParticleBuffer inParticles, unsigned int inNumParticles,
        unsigned int startIndex, int bufferMask);
};