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
    vpm::vec3 X;          // Position
    vpm::vec3 Gamma;      // Vectorial circulation
    vpm::real sigma;     // Smoothing radius
    vpm::pidx_t index;          // Indices of particles
    vpm::vec3 U;          // Velocity at particle
    vpm::mat3 J;          // Jacobian at particle
    vpm::mat3 M;          // Auxiliary memory
    vpm::vec3 C;          // SFS coefficient, numerator, denominator
    vpm::vec3 SFS;

    /*vpm::real vol;            // Volume
    vpm::real circulation;    // Scalar circulation
    bool isStatic;           // Indicates if particle is static
    vpm::vec3 PSE;             // Particle-strength exchange*/

    // Constructor
    Particle()
        : X(0.0f), Gamma(0.0f), sigma(0.0f),
        U(0.0f), J(0.0f), M(0.0f), C(0.0f), SFS(0.0f), index(0) {}
    //PSE(0.0f), vol(0.0f), circulation(0.0f), isStatic(false), 

    __host__ __device__ void reset();    // Reset particle U, J and PSE
    __host__ __device__ void resetSFS(); // Reset particle SFS
};

// ParticleField definition
struct ParticleField
{
    ParticleBuffer particles;        // Host particle buffer
    ParticleBuffer dev_particles;   // Device particle buffer
    vpm::pidx_t numParticles;           // Number of particles in the field
    unsigned int timeStep;               // Current time step
    KernelType kernel;                   // Vortex particle kernel
    vpm::vec3 uInf;               // Uniform freestream function
    std::unique_ptr<SFSScheme> sfs;                      // Subfilter-scale contributions scheme
    std::unique_ptr<RelaxationScheme> relaxation;               // Relaxation scheme
    int synchronized;           // Flags if host buffers are synchronized with device

    // Constructor
    ParticleField(
        ParticleBuffer&& particles,
        vpm::pidx_t numParticles,
        unsigned int timeStep = 0,
        KernelType kernel = KernelType::GAUSSIAN_ERF,
        vpm::vec3 uInf = vpm::vec3(0, 0, 0),
        std::unique_ptr<SFSScheme> sfs = std::make_unique<NoSFS>(),
        std::unique_ptr<RelaxationScheme> relaxation = std::make_unique<PedrizzettiRelaxation>(0.005f)
    );
    // Destructor
    ~ParticleField() {};

	void syncParticlesDeviceToHost(int bufferMask, cudaStream_t stream = 0);
	void syncParticlesHostToDevice(int bufferMask, cudaStream_t stream = 0);

    void addParticleDevice(Particle& particle);
    void overwriteParticleDevice(Particle& particle, vpm::pidx_t index);
    void removeParticleDevice(vpm::pidx_t index);
    void cpyParticlesDeviceToDevice(ParticleBuffer& srcBuffer, vpm::pidx_t srcIndex, vpm::pidx_t count,
        int bufferMask);
};