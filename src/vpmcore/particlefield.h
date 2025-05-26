#pragma once

#include <memory>
#include "common.h"
#include "particlebuffer.h"
#include "kernels.h"
#include "sfs.h"
#include "relaxation.h"

struct OutputType {
    enum Type {
        NONE = 0,
        X = 1 << 0,
        U = 1 << 1,
        OMEGA = 1 << 2,
        GAMMA = 1 << 3,
        SIGMA = 1 << 4,
        INDEX = 1 << 5,
        ALL = 0xFFFF
    };
};

struct Particle {
    vpmvec3 X;          // Position
    vpmvec3 Gamma;      // Vectorial circulation
    vpmfloat sigma;     // Smoothing radius
    int index;          // Indices of particles
    vpmvec3 U;          // Velocity at particle
    vpmmat3 J;          // Jacobian at particle
    vpmmat3 M;          // Auxiliary memory
    vpmvec3 C;          // SFS coefficient, numerator, denominator
    vpmvec3 SFS;

    /*vpmfloat vol;            // Volume
    vpmfloat circulation;    // Scalar circulation
    bool isStatic;           // Indicates if particle is static
    vpmvec3 PSE;             // Particle-strength exchange*/

    // Constructor
    Particle()
        : X(0.0f), Gamma(0.0f), sigma(0.0f),
        U(0.0f), J(0.0f), M(0.0f), C(0.0f), SFS(0.0f), index(0) {}
    //PSE(0.0f), vol(0.0f), circulation(0.0f), isStatic(false), 

    __host__ __device__ void Particle::reset();    // Reset particle U, J and PSE
    __host__ __device__ void Particle::resetSFS(); // Reset particle SFS
};

// ParticleField definition
struct ParticleField {
    unsigned int maxParticles;           // Maximum number of particles
    ParticleBuffer particles;        // Pointer to host particle buffer
    ParticleBuffer dev_particles{ ParticleBufferType::DEVICE }; // Pointer to device particle buffer
    unsigned int numParticles;           // Number of particles in the field
    unsigned int timeStep;               // Current time step
    KernelType kernel;                   // Vortex particle kernel
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
        KernelType kernel = KernelType::GAUSSIAN_ERF,
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