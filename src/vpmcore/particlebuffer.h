#pragma once

#include "common.h"
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
    vpmvec3* X_ = nullptr;          // Position
    vpmvec3* Gamma_ = nullptr;      // Vectorial circulation
    vpmfloat* sigma_ = nullptr;     // Smoothing radius
    size_t* index_ = nullptr;          // Indices of particles
    vpmvec3* U_ = nullptr;          // Velocity at particle
    vpmmat3* J_ = nullptr;          // Jacobian at particle
    vpmmat3* M_ = nullptr;          // Auxiliary memory
    vpmvec3* C_ = nullptr;          // SFS coefficient, numerator, denominator
    vpmvec3* SFS_ = nullptr;

    /*vpmfloat* vol_ = nullptr;         // Volume
    vpmfloat* circulation_ = nullptr;   // Scalar circulation
    bool* isStatic_ = nullptr;          // Indicates if particle is static
    vpmvec3* PSE_ = nullptr;            // Particle-strength exchange*/
    
public:
    const ParticleBufferType::Type bufferType;

    ParticleBuffer(ParticleBufferType::Type bufferType, size_t size) : bufferType(bufferType), count(size) {};

    __host__ __device__ size_t size() const { return count; }
    __host__ __device__ int fields() const { return bufferFields; }
    __host__ __device__ vpmvec3* X() { return X_; }
    __host__ __device__ vpmvec3* Gamma() { return Gamma_; }
    __host__ __device__ vpmfloat* sigma() { return sigma_; }
    __host__ __device__ size_t* index() { return index_; }
    __host__ __device__ vpmvec3* U() { return U_; }
    __host__ __device__ vpmmat3* J() { return J_; }
    __host__ __device__ vpmmat3* M() { return M_; }
    __host__ __device__ vpmvec3* C() { return C_; }
    __host__ __device__ vpmvec3* SFS() { return SFS_; }
	// Const versions
    __host__ __device__ const vpmvec3* X() const { return X_; }
    __host__ __device__ const vpmvec3* Gamma() const { return Gamma_; }
    __host__ __device__ const vpmfloat* sigma() const { return sigma_; }
    __host__ __device__ const size_t* index() const { return index_; }
    __host__ __device__ const vpmvec3* U() const { return U_; }
    __host__ __device__ const vpmmat3* J() const { return J_; }
    __host__ __device__ const vpmmat3* M() const { return M_; }
    __host__ __device__ const vpmvec3* C() const { return C_; }
    __host__ __device__ const vpmvec3* SFS() const { return SFS_; }
    
	/*vpmfloat* vol() { return vol_; }
	vpmfloat* circulation() { return circulation_; }
	bool* isStatic() { return isStatic_; }
	vpmvec3* PSE() { return PSE_; }*/

    void permute(std::span<const size_t> indices, int bufferMask);
    void mallocFields(int bufferMask);
    void freeFields();
    void freeFields(int bufferMask);
};

size_t cpyParticleBuffer(ParticleBuffer dstBuffer, ParticleBuffer srcBuffer, 
    size_t dstIndex, size_t srcIndex, size_t count, int bufferMask,  cudaStream_t stream = 0);

size_t cpyParticleBuffer(ParticleBuffer dstBuffer, ParticleBuffer srcBuffer, int bufferMask,
    cudaStream_t stream = 0);