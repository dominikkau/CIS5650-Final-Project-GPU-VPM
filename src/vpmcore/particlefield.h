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
    vpm::vec3 X_;          // Position
    vpm::real GammaX_;      // Vectorial circulation
    vpm::real GammaY_;
    vpm::real GammaZ_;
    vpm::real sigma_;     // Smoothing radius
    vpm::pidx_t index_;          // Indices of particles
    vpm::vec3 U_;          // Velocity at particle
    vpm::mat3 J_;          // Jacobian at particle
    vpm::mat3 M_;          // Auxiliary memory
    vpm::vec3 C_;          // SFS coefficient, numerator, denominator
    vpm::vec3 SFS_;

    /*vpm::real vol;            // Volume
    vpm::real circulation;    // Scalar circulation
    bool isStatic;           // Indicates if particle is static
    vpm::vec3 PSE;             // Particle-strength exchange*/

    // Constructor
    Particle()
        : X_(0.0f), GammaX_(0.0f), GammaY_(0.0f), GammaZ_(0.0f), sigma_(0.0f),
        U_(0.0f), J_(0.0f), M_(0.0f), C_(0.0f), SFS_(0.0f), index_(0) {}
    //PSE(0.0f), vol(0.0f), circulation(0.0f), isStatic(false), 

    __host__ __device__ void reset();    // Reset particle U, J and PSE
    __host__ __device__ void resetSFS(); // Reset particle SFS

    __host__ __device__ auto* X() { return &X_; }
    __host__ __device__ auto* GammaX() { return &GammaX_; }
    __host__ __device__ auto* GammaY() { return &GammaY_; }
    __host__ __device__ auto* GammaZ() { return &GammaZ_; }
    __host__ __device__ auto* sigma() { return &sigma_; }
    __host__ __device__ auto* index() { return &index_; }
    __host__ __device__ auto* U() { return &U_; }
    __host__ __device__ auto* J() { return &J_; }
    __host__ __device__ auto* M() { return &M_; }
    __host__ __device__ auto* C() { return &C_; }
    __host__ __device__ auto* SFS() { return &SFS_; }
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