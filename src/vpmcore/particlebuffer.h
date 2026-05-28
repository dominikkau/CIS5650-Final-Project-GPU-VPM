#pragma once

#include "common.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <span>

struct BufferField {
    enum Type {
        NONE = 0,
        X = 1 << 0,
        U = 1 << 1,
        J = 1 << 2,
        GAMMA = 1 << 3,
        SIGMA = 1 << 4,
        SFS = 1 << 5,
        C = 1 << 6,
        M = 1 << 7,
        INDEX = 1 << 8,
        PSE = 1 << 9,
        VOL = 1 << 10,
        CIRCULATION = 1 << 11,
        ISSTATIC = 1 << 12,
        ALL = 0xFFFF
        // Add other buffers as needed
    };
};

struct ParticleBufferType {
    enum Type {
        DEVICE,
        HOST,
        HOST_PINNED
    };
};

class ParticleBuffer {
	size_t count;                  // Maximum number of particles in the buffer
    int bufferFields = 0;
    vpm::vec3* X_ = nullptr;          // Position
    vpm::vec3* Gamma_ = nullptr;      // Vectorial circulation
    vpm::real* sigma_ = nullptr;     // Smoothing radius
    size_t* index_ = nullptr;          // Indices of particles
    vpm::vec3* U_ = nullptr;          // Velocity at particle
    vpm::mat3* J_ = nullptr;          // Jacobian at particle
    vpm::mat3* M_ = nullptr;          // Auxiliary memory
    vpm::vec3* C_ = nullptr;          // SFS coefficient, numerator, denominator
    vpm::vec3* SFS_ = nullptr;

    /*vpm::real* vol_ = nullptr;         // Volume
    vpm::real* circulation_ = nullptr;   // Scalar circulation
    bool* isStatic_ = nullptr;          // Indicates if particle is static
    vpm::vec3* PSE_ = nullptr;            // Particle-strength exchange*/
    
public:
    const ParticleBufferType::Type bufferType;

    ParticleBuffer(ParticleBufferType::Type bufferType, size_t size) : bufferType(bufferType), count(size) {};
    ~ParticleBuffer() { freeFields(); }

    __host__ __device__ size_t size() const { return count; }
    __host__ __device__ int fields() const { return bufferFields; }
    __host__ __device__ vpm::vec3* X() { return X_; }
    __host__ __device__ vpm::vec3* Gamma() { return Gamma_; }
    __host__ __device__ vpm::real* sigma() { return sigma_; }
    __host__ __device__ size_t* index() { return index_; }
    __host__ __device__ vpm::vec3* U() { return U_; }
    __host__ __device__ vpm::mat3* J() { return J_; }
    __host__ __device__ vpm::mat3* M() { return M_; }
    __host__ __device__ vpm::vec3* C() { return C_; }
    __host__ __device__ vpm::vec3* SFS() { return SFS_; }

	// Const versions
    __host__ __device__ const vpm::vec3* X() const { return X_; }
    __host__ __device__ const vpm::vec3* Gamma() const { return Gamma_; }
    __host__ __device__ const vpm::real* sigma() const { return sigma_; }
    __host__ __device__ const size_t* index() const { return index_; }
    __host__ __device__ const vpm::vec3* U() const { return U_; }
    __host__ __device__ const vpm::mat3* J() const { return J_; }
    __host__ __device__ const vpm::mat3* M() const { return M_; }
    __host__ __device__ const vpm::vec3* C() const { return C_; }
    __host__ __device__ const vpm::vec3* SFS() const { return SFS_; }
    
	/*vpm::real* vol() { return vol_; }
	vpm::real* circulation() { return circulation_; }
	bool* isStatic() { return isStatic_; }
	vpm::vec3* PSE() { return PSE_; }*/

    void permute(std::span<const size_t> indices, int bufferMask);
    void mallocFields(int bufferMask);
    void freeFields();
    void freeFields(int bufferMask);
};

size_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer, 
    size_t dstIndex, size_t srcIndex, size_t count, int bufferMask,  cudaStream_t stream = 0);

size_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer, int bufferMask,
    cudaStream_t stream = 0);