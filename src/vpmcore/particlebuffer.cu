#include <cuda.h>
#include <cuda_runtime.h>
#include "particlebuffer.h"
#include <vector>
#include <iostream>

ParticleBuffer::ParticleBuffer(ParticleBuffer&& other) noexcept
	: type_(other.type_),
      size_(other.size_),
      fields_(other.fields_),
      X_(other.X_),
      Gamma_(other.Gamma_),
      sigma_(other.sigma_),
      index_(other.index_),
      U_(other.U_),
      J_(other.J_),
      M_(other.M_),
      C_(other.C_),
      SFS_(other.SFS_)
{
	other.X_ = nullptr;
	other.Gamma_ = nullptr;
	other.sigma_ = nullptr;
	other.index_ = nullptr;
	other.U_ = nullptr;
	other.J_ = nullptr;
	other.M_ = nullptr;
	other.C_ = nullptr;
	other.SFS_ = nullptr;
    other.fields_ = 0;
}

ParticleBuffer& ParticleBuffer::operator=(ParticleBuffer&& other) noexcept
{
	if (this == &other) return *this;

    freeFields();

    forEachFieldPair(
        other,
        [](auto*& ptrA, auto* ptrB) { ptrA = ptrB; },
        other.fields_
	);
    type_ = other.type_;
    size_ = other.size_;
    fields_ = other.fields_;

	other.forEachField(
		[this](auto*& ptr) { ptr = nullptr; }
	);
	other.fields_ = 0;
	
	return *this;
}

void ParticleBuffer::mallocFields(int bufferMask)
{
    // Ignore fields that have been allocated
	bufferMask &= ~fields_;

    switch (type_)
    {
    case ParticleBufferType::DEVICE:
        forEachField(
            [this](auto*& ptr)
            {
                cudaMalloc(&ptr, size_ * sizeof(*ptr));
                checkCUDAError((std::string("cudaMalloc failed for type ") + typeid(*ptr).name()).c_str());
            }, bufferMask
        );
        break;
    case ParticleBufferType::HOST:
        forEachField(
            [this](auto*& ptr)
            {
                using ElementType = std::remove_cvref_t<decltype(*ptr)>;
                ptr = new ElementType[size_];
            }, bufferMask
        );
        break;
    case ParticleBufferType::HOST_PINNED:
        forEachField(
            [this](auto*& ptr)
            {
                cudaMallocHost(&ptr, size_ * sizeof(*ptr));
                checkCUDAError((std::string("cudaMallocHost failed for type ") + typeid(*ptr).name()).c_str());
            }, bufferMask
        );
        break;
    }
    
    // Update allocated fields
    fields_ |= bufferMask;
}

void ParticleBuffer::freeFields(int bufferMask)
{
    // Ignore fields that have not been allocated
    bufferMask &= fields_;

    switch (type_)
    {
    case ParticleBufferType::DEVICE:
        forEachField(
            [this](auto*& ptr)
            {
                cudaFree(ptr);
                checkCUDAError((std::string("cudaFree failed for type ") + typeid(*ptr).name()).c_str());
                ptr = nullptr;
            }, bufferMask
        );
        break;
    case ParticleBufferType::HOST:
        forEachField(
            [this](auto*& ptr)
            {
                delete[] ptr;
                ptr = nullptr;
            }, bufferMask
        );
        break;
    case ParticleBufferType::HOST_PINNED:
        forEachField(
            [this](auto*& ptr)
            {
                cudaFreeHost(ptr);
                checkCUDAError((std::string("cudaFreeHost failed for type ") + typeid(*ptr).name()).c_str());
                ptr = nullptr;
            }, bufferMask
        );
        break;
    }

    // Update allocated fields
    fields_ &= ~bufferMask;
}

// Frees memory for all allocated fields
void ParticleBuffer::freeFields() {
    freeFields(fields_);
}

void ParticleBuffer::permute(std::span<const vpm::pidx_t> indices, int bufferMask)
{
	if (indices.size() != size_) {
		std::cerr << "Permutation indices size does not match particle size_" << std::endl;
		return;
	}

    // Ignore unallocated fields
    bufferMask &= fields_;

    std::vector<bool> visited(size_, false);

    for (size_t i = 0; i < size_; ++i) {
        if (visited[i]) continue;
        if (indices[i] == i)
        { 
            visited[i] = true;
            continue;
        }
        vpm::pidx_t j = static_cast<vpm::pidx_t>(i);
        while (!visited[j]) {
            visited[j] = true;
            vpm::pidx_t next = indices[j];
            if (!visited[next]) {
                forEachField(
                    [j, next](auto* ptr) { std::swap(ptr[j], ptr[next]); }, bufferMask
                );
            }
            j = next;
        }
    }
}

vpm::pidx_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer,
    int bufferMask, cudaStream_t stream)
{
	return cpyParticleBuffer(dstBuffer, srcBuffer, 0, 0, std::min(dstBuffer.size(), srcBuffer.size()), bufferMask, stream);
}

cudaMemcpyKind copyDirection(const ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer)
{
    if (dstBuffer.type() == ParticleBufferType::DEVICE) {
        if (srcBuffer.type() == ParticleBufferType::DEVICE)
            return cudaMemcpyDeviceToDevice;
        else                                                
            return cudaMemcpyHostToDevice;
    }
    else {
        if (srcBuffer.type() == ParticleBufferType::DEVICE)
            return cudaMemcpyDeviceToHost;
        else                                                
            return cudaMemcpyHostToHost;
    }
}

vpm::pidx_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer,
    vpm::pidx_t dstIndex, vpm::pidx_t srcIndex, vpm::pidx_t count, int bufferMask, cudaStream_t stream) {

    // Determine cudaMemcpy direction
	auto cpyDirection = copyDirection(dstBuffer, srcBuffer);

	// Check for out-of-bounds indices
    if (srcIndex > srcBuffer.size()) return 0;
	if (dstIndex > dstBuffer.size()) return 0;

	// Adjust size_ to prevent out-of-bounds access
    count = std::min(count, srcBuffer.size() - srcIndex);
    count = std::min(count, dstBuffer.size() - dstIndex);

    // Ensure that we do not try to copy from or to non-existent fields
    bufferMask &= (dstBuffer.fields() & srcBuffer.fields());

    dstBuffer.forEachFieldPair(srcBuffer,
        [dstIndex, srcIndex, count, cpyDirection, stream](auto* ptrA, const auto* ptrB)
        {
            cudaMemcpyAsync(ptrA + dstIndex, ptrB + srcIndex, count * sizeof(*ptrA), cpyDirection, stream);
		}, bufferMask
    );

    return count;
}