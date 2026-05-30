#pragma once

#include "common.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <span>
#include <string>

struct BufferField
{
    enum Type
    {
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

struct ParticleBufferType
{
    enum Type
    {
        DEVICE,
        HOST,
        HOST_PINNED
    };
};

class ParticleBuffer
{
    ParticleBufferType::Type type_;
    vpm::pidx_t size_;                  // Maximum number of particles in the buffer
    int fields_ = 0;
    vpm::vec3* X_ = nullptr;          // Position
	vpm::real* GammaX_ = nullptr;     // Scalar circulation in x-direction
	vpm::real* GammaY_ = nullptr;     // Scalar circulation in y-direction
	vpm::real* GammaZ_ = nullptr;     // Scalar circulation in z-direction
    vpm::real* sigma_ = nullptr;     // Smoothing radius
    vpm::pidx_t* index_ = nullptr;          // Indices of particles
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
    ParticleBuffer(ParticleBufferType::Type type, vpm::pidx_t size) : type_(type), size_(size) {};
    ~ParticleBuffer() { freeFields(); }

    ParticleBuffer(const ParticleBuffer& other) = delete;
    ParticleBuffer& operator=(const ParticleBuffer& other) = delete;

    ParticleBuffer(ParticleBuffer&& other) noexcept;
    ParticleBuffer& operator=(ParticleBuffer&& other) noexcept;

    __host__ __device__ auto size() const { return size_; }
    __host__ __device__ auto fields() const { return fields_; }
	__host__ __device__ auto type() const { return type_; }

    __host__ __device__ auto* X() { return X_; }
    __host__ __device__ auto* GammaX() { return GammaX_; }
    __host__ __device__ auto* GammaY() { return GammaY_; }
    __host__ __device__ auto* GammaZ() { return GammaZ_; }
    __host__ __device__ auto* sigma() { return sigma_; }
    __host__ __device__ auto* index() { return index_; }
    __host__ __device__ auto* U() { return U_; }
    __host__ __device__ auto* J() { return J_; }
    __host__ __device__ auto* M() { return M_; }
    __host__ __device__ auto* C() { return C_; }
    __host__ __device__ auto* SFS() { return SFS_; }

    __host__ __device__ auto& X(vpm::pidx_t i) { return X_[i]; }
    __host__ __device__ auto& GammaX(vpm::pidx_t i) { return GammaX_[i]; }
    __host__ __device__ auto& GammaY(vpm::pidx_t i) { return GammaY_[i]; }
    __host__ __device__ auto& GammaZ(vpm::pidx_t i) { return GammaZ_[i]; }
    __host__ __device__ auto& sigma(vpm::pidx_t i) { return sigma_[i]; }
    __host__ __device__ auto& index(vpm::pidx_t i) { return index_[i]; }
    __host__ __device__ auto& U(vpm::pidx_t i) { return U_[i]; }
    __host__ __device__ auto& J(vpm::pidx_t i) { return J_[i]; }
    __host__ __device__ auto& M(vpm::pidx_t i) { return M_[i]; }
    __host__ __device__ auto& C(vpm::pidx_t i) { return C_[i]; }
    __host__ __device__ auto& SFS(vpm::pidx_t i) { return SFS_[i]; }

    // Const versions
    __host__ __device__ const auto* X() const { return X_; }
    __host__ __device__ const auto* GammaX() const { return GammaX_; }
    __host__ __device__ const auto* GammaY() const { return GammaY_; }
    __host__ __device__ const auto* GammaZ() const { return GammaZ_; }
    __host__ __device__ const auto* sigma() const { return sigma_; }
    __host__ __device__ const auto* index() const { return index_; }
    __host__ __device__ const auto* U() const { return U_; }
    __host__ __device__ const auto* J() const { return J_; }
    __host__ __device__ const auto* M() const { return M_; }
    __host__ __device__ const auto* C() const { return C_; }
    __host__ __device__ const auto* SFS() const { return SFS_; }

    __host__ __device__ const auto& X(vpm::pidx_t i) const { return X_[i]; }
    __host__ __device__ const auto& GammaX(vpm::pidx_t i) const { return GammaX_[i]; }
    __host__ __device__ const auto& GammaY(vpm::pidx_t i) const { return GammaY_[i]; }
    __host__ __device__ const auto& GammaZ(vpm::pidx_t i) const { return GammaZ_[i]; }
    __host__ __device__ const auto& sigma(vpm::pidx_t i) const { return sigma_[i]; }
    __host__ __device__ const auto& index(vpm::pidx_t i) const { return index_[i]; }
    __host__ __device__ const auto& U(vpm::pidx_t i) const { return U_[i]; }
    __host__ __device__ const auto& J(vpm::pidx_t i) const { return J_[i]; }
    __host__ __device__ const auto& M(vpm::pidx_t i) const { return M_[i]; }
    __host__ __device__ const auto& C(vpm::pidx_t i) const { return C_[i]; }
    __host__ __device__ const auto& SFS(vpm::pidx_t i) const { return SFS_[i]; }

    template <typename Fun>
    void forEachField(Fun&& fun, int mask = BufferField::ALL)
    {
        if (mask & BufferField::X)      fun(X_);
        if (mask & BufferField::U)      fun(U_);
        if (mask & BufferField::J)      fun(J_);
		if (mask & BufferField::GAMMA)  fun(GammaX_);
        if (mask & BufferField::GAMMA)  fun(GammaY_);
        if (mask & BufferField::GAMMA)  fun(GammaZ_);
        if (mask & BufferField::SIGMA)  fun(sigma_);
        if (mask & BufferField::SFS)    fun(SFS_);
        if (mask & BufferField::C)      fun(C_);
        if (mask & BufferField::M)      fun(M_);
        if (mask & BufferField::INDEX)  fun(index_);
    }

    template <typename BufferType, typename Fun>
    void forEachFieldPair(BufferType&& buffer, Fun&& fun, int mask = BufferField::ALL)
    {
        if (mask & BufferField::X)      fun(X_,      buffer.X());
        if (mask & BufferField::U)      fun(U_,      buffer.U());
        if (mask & BufferField::J)      fun(J_,      buffer.J());
        if (mask & BufferField::GAMMA)  fun(GammaX_, buffer.GammaX());
        if (mask & BufferField::GAMMA)  fun(GammaY_, buffer.GammaY());
        if (mask & BufferField::GAMMA)  fun(GammaZ_, buffer.GammaZ());
        if (mask & BufferField::SIGMA)  fun(sigma_,  buffer.sigma());
        if (mask & BufferField::SFS)    fun(SFS_,    buffer.SFS());
        if (mask & BufferField::C)      fun(C_,      buffer.C());
        if (mask & BufferField::M)      fun(M_,      buffer.M());
        if (mask & BufferField::INDEX)  fun(index_,  buffer.index());
    }

    //template <typename Fun>
    //void forEachFieldPair(ParticleBuffer& buffer, Fun&& fun, int mask = BufferField::ALL)
    //{
    //    if (mask & BufferField::X)      fun(X_,      buffer.X());
    //    if (mask & BufferField::U)      fun(U_,      buffer.U());
    //    if (mask & BufferField::J)      fun(J_,      buffer.J());
    //    if (mask & BufferField::GAMMA)  fun(GammaX_, buffer.GammaX());
    //    if (mask & BufferField::GAMMA)  fun(GammaY_, buffer.GammaY());
    //    if (mask & BufferField::GAMMA)  fun(GammaZ_, buffer.GammaZ());
    //    if (mask & BufferField::SIGMA)  fun(sigma_,  buffer.sigma());
    //    if (mask & BufferField::SFS)    fun(SFS_,    buffer.SFS());
    //    if (mask & BufferField::C)      fun(C_,      buffer.C());
    //    if (mask & BufferField::M)      fun(M_,      buffer.M());
    //    if (mask & BufferField::INDEX)  fun(index_,  buffer.index());
    //}

    //template <typename Fun>
    //void forEachFieldPair(Particle& particle, Fun&& fun, int mask = BufferField::ALL)
    //{
    //    if (mask & BufferField::X)      fun(X_,      particle.X());
    //    if (mask & BufferField::U)      fun(U_,      particle.U());
    //    if (mask & BufferField::J)      fun(J_,      particle.J());
    //    if (mask & BufferField::GAMMA)  fun(GammaX_, particle.GammaX()); fun(GammaY_, particle.GammaY()); fun(GammaZ_, particle.GammaZ());
    //    if (mask & BufferField::SIGMA)  fun(sigma_,  particle.sigma());
    //    if (mask & BufferField::SFS)    fun(SFS_,    particle.SFS());
    //    if (mask & BufferField::C)      fun(C_,      particle.C());
    //    if (mask & BufferField::M)      fun(M_,      particle.M());
    //    if (mask & BufferField::INDEX)  fun(index_,  particle.index());
    //}

    void permute(std::span<const vpm::pidx_t> indices, int bufferMask);
    void mallocFields(int bufferMask);
    void freeFields();
    void freeFields(int bufferMask);
};

vpm::pidx_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer,
    vpm::pidx_t dstIndex, vpm::pidx_t srcIndex, vpm::pidx_t count, int bufferMask,  cudaStream_t stream = 0);

vpm::pidx_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer,
    int bufferMask, cudaStream_t stream = 0);

// Leightweight struct to pass pointers to kernels
struct ParticleBufferView
{
    const vpm::pidx_t size;
    vpm::vec3* const X;
    vpm::real* const GammaX;
    vpm::real* const GammaY;
    vpm::real* const GammaZ;
    vpm::real* const sigma;
    vpm::pidx_t* const index;
    vpm::vec3* const U;
    vpm::mat3* const J;
    vpm::mat3* const M;
    vpm::vec3* const C;
    vpm::vec3* const SFS;

    ParticleBufferView(ParticleBuffer& buffer)
        : size(buffer.size()), X(buffer.X()), GammaX(buffer.GammaX()), GammaY(buffer.GammaY()), GammaZ(buffer.GammaZ()),
        sigma(buffer.sigma()), index(buffer.index()), U(buffer.U()), J(buffer.J()), M(buffer.M()),
        C(buffer.C()), SFS(buffer.SFS()) {
    }
};