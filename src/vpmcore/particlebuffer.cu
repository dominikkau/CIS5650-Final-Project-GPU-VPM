#include <cuda.h>
#include <cuda_runtime.h>
#include "particlebuffer.h"
#include <vector>
#include <iostream>

// Allocates memory for fields in bufferMask
// Ignores fields that have been allocated already
void ParticleBuffer::mallocFields(int bufferMask) {
    // Ignore fields that have been allocated
    bufferMask &= ~bufferFields;

    switch (bufferType) {
    case ParticleBufferType::DEVICE:
        if (bufferMask & BufferField::X) {
            cudaMalloc((void**)&X_, count * sizeof(vpm::vec3));
            checkCUDAError("cudaMalloc of X failed!");
        }

        if (bufferMask & BufferField::U) {
            cudaMalloc((void**)&U_, count * sizeof(vpm::vec3));
            checkCUDAError("cudaMalloc of U failed!");
        }

        if (bufferMask & BufferField::J) {
            cudaMalloc((void**)&J_, count * sizeof(vpm::mat3));
            checkCUDAError("cudaMalloc of J failed!");
        }

        if (bufferMask & BufferField::GAMMA) {
            cudaMalloc((void**)&Gamma_, count * sizeof(vpm::vec3));
            checkCUDAError("cudaMalloc of Gamma failed!");
        }

        if (bufferMask & BufferField::SIGMA) {
            cudaMalloc((void**)&sigma_, count * sizeof(vpm::real));
            checkCUDAError("cudaMalloc of sigma failed!");
        }

        if (bufferMask & BufferField::SFS) {
            cudaMalloc((void**)&SFS_, count * sizeof(vpm::vec3));
            checkCUDAError("cudaMalloc of SFS failed!");
        }

        if (bufferMask & BufferField::C) {
            cudaMalloc((void**)&C_, count * sizeof(vpm::vec3));
            checkCUDAError("cudaMalloc of C failed!");
        }

        if (bufferMask & BufferField::M) {
            cudaMalloc((void**)&M_, count * sizeof(vpm::mat3));
            checkCUDAError("cudaMalloc of M failed!");
        }

        if (bufferMask & BufferField::INDEX) {
            cudaMalloc((void**)&index_, count * sizeof(vpm::pidx_t));
            checkCUDAError("cudaMalloc of index failed!");
        }

        /*if (bufferMask & BufferField::PSE) {
            cudaMalloc((void**)&PSE_, count * sizeof(vpm::vec3));
            checkCUDAError("cudaMalloc of PSE failed!");
        }

        if (bufferMask & BufferField::IS_STATIC) {
            cudaMalloc((void**)&isStatic_, count * sizeof(bool));
            checkCUDAError("cudaMalloc of isStatic failed!");
        }

        if (bufferMask & BufferField::VOL) {
            cudaMalloc((void**)&vol_, count * sizeof(vpm::real));
            checkCUDAError("cudaMalloc of vol failed!");
        }

        if (bufferMask & BufferField::CIRC) {
            cudaMalloc((void**)&circulation_, count * sizeof(vpm::real));
            checkCUDAError("cudaMalloc of circulation failed!");
        }*/
        break;

    case ParticleBufferType::HOST:
        if (bufferMask & BufferField::X) X_ = new vpm::vec3[count];
        if (bufferMask & BufferField::U) U_ = new vpm::vec3[count];
        if (bufferMask & BufferField::J) J_ = new vpm::mat3[count];
        if (bufferMask & BufferField::GAMMA) Gamma_ = new vpm::vec3[count];
        if (bufferMask & BufferField::SIGMA) sigma_ = new vpm::real[count];
        if (bufferMask & BufferField::SFS) SFS_ = new vpm::vec3[count];
        if (bufferMask & BufferField::C) C_ = new vpm::vec3[count];
        if (bufferMask & BufferField::M) M_ = new vpm::mat3[count];
        if (bufferMask & BufferField::INDEX) index_ = new vpm::pidx_t[count];
        /*if (bufferMask & BufferField::PSE) PSE_ = new vpm::vec3[count];
        if (bufferMask & BufferField::IS_STATIC) isStatic_ = new bool[count];
        if (bufferMask & BufferField::VOL) vol_ = new vpm::real[count];
        if (bufferMask & BufferField::CIRC) circulation_ = new vpm::real[count];*/
        break;

    case ParticleBufferType::HOST_PINNED:
        if (bufferMask & BufferField::X) cudaMallocHost((void**)&X_, count * sizeof(vpm::vec3));
        if (bufferMask & BufferField::U) cudaMallocHost((void**)&U_, count * sizeof(vpm::vec3));
        if (bufferMask & BufferField::J) cudaMallocHost((void**)&J_, count * sizeof(vpm::mat3));
        if (bufferMask & BufferField::GAMMA) cudaMallocHost((void**)&Gamma_, count * sizeof(vpm::vec3));
        if (bufferMask & BufferField::SIGMA) cudaMallocHost((void**)&sigma_, count * sizeof(vpm::real));
        if (bufferMask & BufferField::SFS) cudaMallocHost((void**)&SFS_, count * sizeof(vpm::vec3));
        if (bufferMask & BufferField::C) cudaMallocHost((void**)&C_, count * sizeof(vpm::vec3));
        if (bufferMask & BufferField::M) cudaMallocHost((void**)&M_, count * sizeof(vpm::mat3));
        if (bufferMask & BufferField::INDEX) cudaMallocHost((void**)&index_, count * sizeof(vpm::pidx_t));
        /*if (bufferMask & BufferField::PSE) cudaMallocHost((void**)&PSE_, count * sizeof(vpm::vec3));
        if (bufferMask & BufferField::IS_STATIC) cudaMallocHost((void**)&isStatic_, count * sizeof(bool));
        if (bufferMask & BufferField::VOL) cudaMallocHost((void**)&vol_, count * sizeof(vpm::real));
        if (bufferMask & BufferField::CIRC) cudaMallocHost((void**)&circulation_, count * sizeof(vpm::real));*/
        break;
    }

    // Update allocated fields
    bufferFields |= bufferMask;
}

void ParticleBuffer::permute(std::span<const vpm::pidx_t> indices, int bufferMask)
{
	if (indices.size() != count) {
		std::cerr << "Permutation indices size does not match particle count" << std::endl;
		return;
	}

    // Ignore unallocated fields
    bufferMask &= bufferFields;

    std::vector<bool> visited(count, false);

    for (size_t i = 0; i < count; ++i) {
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
                if (bufferMask & BufferField::X) std::swap(X_[j], X_[next]);
                if (bufferMask & BufferField::U) std::swap(U_[j], U_[next]);
                if (bufferMask & BufferField::J) std::swap(J_[j], J_[next]);
                if (bufferMask & BufferField::GAMMA) std::swap(Gamma_[j], Gamma_[next]);
                if (bufferMask & BufferField::SIGMA) std::swap(sigma_[j], sigma_[next]);
                if (bufferMask & BufferField::SFS) std::swap(SFS_[j], SFS_[next]);
                if (bufferMask & BufferField::C) std::swap(C_[j], C_[next]);
                if (bufferMask & BufferField::M) std::swap(M_[j], M_[next]);
                if (bufferMask & BufferField::INDEX) std::swap(index_[j], index_[next]);
                /*if (bufferMask & BufferField::PSE) std::swap(PSE_[j], PSE_[next]);
                if (bufferMask & BufferField::IS_STATIC) std::swap(isStatic_[j], isStatic_[next]);
                if (bufferMask & BufferField::VOL) std::swap(vol_[j], vol_[next]);
                if (bufferMask & BufferField::CIRC) std::swap(circulation_[j], circulation_[next]);*/
            }
            j = next;
        }
    }
}

// Frees memory for all allocated fields
void ParticleBuffer::freeFields() {
	freeFields(bufferFields);
}

// Frees memory for fields in bufferMask
// Ignores fields that have not been allocated
void ParticleBuffer::freeFields(int bufferMask) {
    // Ignore unallocated fields
    bufferMask &= bufferFields;

    switch (bufferType) {
    case ParticleBufferType::DEVICE:
        if (bufferMask & BufferField::X) cudaFree(X_);
        if (bufferMask & BufferField::U) cudaFree(U_);
        if (bufferMask & BufferField::J) cudaFree(J_);
        if (bufferMask & BufferField::GAMMA) cudaFree(Gamma_);
        if (bufferMask & BufferField::SIGMA) cudaFree(sigma_);
        if (bufferMask & BufferField::SFS) cudaFree(SFS_);
        if (bufferMask & BufferField::C) cudaFree(C_);
        if (bufferMask & BufferField::M) cudaFree(M_);
        if (bufferMask & BufferField::INDEX) cudaFree(index_);
        /*if (bufferMask & BufferField::PSE) cudaFree(PSE_);
        if (bufferMask & BufferField::IS_STATIC) cudaFree(isStatic_);
        if (bufferMask & BufferField::VOL) cudaFree(vol_);
        if (bufferMask & BufferField::CIRC) cudaFree(circulation_);*/
        break;

    case ParticleBufferType::HOST:
        if (bufferMask & BufferField::X) delete[] X_;
        if (bufferMask & BufferField::U) delete[] U_;
        if (bufferMask & BufferField::J) delete[] J_;
        if (bufferMask & BufferField::GAMMA) delete[] Gamma_;
        if (bufferMask & BufferField::SIGMA) delete[] sigma_;
        if (bufferMask & BufferField::SFS) delete[] SFS_;
        if (bufferMask & BufferField::C) delete[] C_;
        if (bufferMask & BufferField::M) delete[] M_;
        if (bufferMask & BufferField::INDEX) delete[] index_;
        /*if (bufferMask & BufferField::PSE) delete[] PSE_;
        if (bufferMask & BufferField::IS_STATIC) delete[] isStatic_;
        if (bufferMask & BufferField::VOL) delete[] vol_;
        if (bufferMask & BufferField::CIRC) delete[] circulation_;*/
        break;

    case ParticleBufferType::HOST_PINNED:
        if (bufferMask & BufferField::X) cudaFreeHost(X_);
        if (bufferMask & BufferField::U) cudaFreeHost(U_);
        if (bufferMask & BufferField::J) cudaFreeHost(J_);
        if (bufferMask & BufferField::GAMMA) cudaFreeHost(Gamma_);
        if (bufferMask & BufferField::SIGMA) cudaFreeHost(sigma_);
        if (bufferMask & BufferField::SFS) cudaFreeHost(SFS_);
        if (bufferMask & BufferField::C) cudaFreeHost(C_);
        if (bufferMask & BufferField::M) cudaFreeHost(M_);
        if (bufferMask & BufferField::INDEX) cudaFreeHost(index_);
        /*if (bufferMask & BufferField::PSE) cudaFreeHost(PSE_);
        if (bufferMask & BufferField::IS_STATIC) cudaFreeHost(isStatic_);
        if (bufferMask & BufferField::VOL) cudaFreeHost(vol_);
        if (bufferMask & BufferField::CIRC) cudaFreeHost(circulation_);*/
        break;
    }

    // Update allocated fields
    bufferFields &= ~bufferMask;
}

vpm::pidx_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer, int bufferMask,
    cudaStream_t stream)
{
	return cpyParticleBuffer(dstBuffer, srcBuffer, 0, 0, std::min(dstBuffer.size(), srcBuffer.size()), bufferMask, stream);
}

vpm::pidx_t cpyParticleBuffer(ParticleBuffer& dstBuffer, const ParticleBuffer& srcBuffer,
    vpm::pidx_t dstIndex, vpm::pidx_t srcIndex, vpm::pidx_t count, int bufferMask, cudaStream_t stream) {

    // Determine cudaMemcpy direction
    cudaMemcpyKind cpyDirection;
    if (dstBuffer.bufferType == ParticleBufferType::DEVICE) {
        if (srcBuffer.bufferType == ParticleBufferType::DEVICE) cpyDirection = cudaMemcpyDeviceToDevice;
        else cpyDirection = cudaMemcpyHostToDevice;
    }
    else {
        if (srcBuffer.bufferType == ParticleBufferType::DEVICE) cpyDirection = cudaMemcpyDeviceToHost;
        else cpyDirection = cudaMemcpyHostToHost;
    }

	// Check for out-of-bounds indices
    if (srcIndex > srcBuffer.size()) return 0;
	if (dstIndex > dstBuffer.size()) return 0;

	// Adjust count to prevent out-of-bounds access
	count = std::min(count, srcBuffer.size() - srcIndex);
	count = std::min(count, dstBuffer.size() - dstIndex);

    // Ensure that we do not try to copy from or to non-existent fields
    bufferMask &= (dstBuffer.fields() & srcBuffer.fields());

    if (bufferMask & BufferField::X) {
        cudaMemcpyAsync(dstBuffer.X() + dstIndex, srcBuffer.X(), count * sizeof(vpm::vec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::U) {
        cudaMemcpyAsync(dstBuffer.U() + dstIndex, srcBuffer.U(), count * sizeof(vpm::vec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::J) {
        cudaMemcpyAsync(dstBuffer.J() + dstIndex, srcBuffer.J(), count * sizeof(vpm::mat3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::GAMMA) {
        cudaMemcpyAsync(dstBuffer.Gamma() + dstIndex, srcBuffer.Gamma(), count * sizeof(vpm::vec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::SIGMA) {
        cudaMemcpyAsync(dstBuffer.sigma() + dstIndex, srcBuffer.sigma(), count * sizeof(vpm::real), cpyDirection, stream);
    }
    if (bufferMask & BufferField::SFS) {
        cudaMemcpyAsync(dstBuffer.SFS() + dstIndex, srcBuffer.SFS(), count * sizeof(vpm::vec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::C) {
        cudaMemcpyAsync(dstBuffer.C() + dstIndex, srcBuffer.C(), count * sizeof(vpm::vec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::M) {
        cudaMemcpyAsync(dstBuffer.M() + dstIndex, srcBuffer.M(), count * sizeof(vpm::mat3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::INDEX) {
        cudaMemcpyAsync(dstBuffer.index() + dstIndex, srcBuffer.index(), count * sizeof(vpm::pidx_t), cpyDirection, stream);
    }
    /*if (bufferMask & BufferField::PSE) {
        cudaMemcpyAsync(dstBuffer.PSE() + dstIndex, srcBuffer.PSE(), count * sizeof(vpm::vec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::IS_STATIC) {
        cudaMemcpyAsync(dstBuffer.isStatic() + dstIndex, srcBuffer.isStatic(), count * sizeof(bool), cpyDirection, stream);
    }
    if (bufferMask & BufferField::VOL) {
        cudaMemcpyAsync(dstBuffer.vol() + dstIndex, srcBuffer.vol(), count * sizeof(vpm::real), cpyDirection, stream);
    }
    if (bufferMask & BufferField::CIRC) {
        cudaMemcpyAsync(dstBuffer.circulation() + dstIndex, srcBuffer.circulation(), count * sizeof(vpm::real), cpyDirection, stream);
    }*/

    return count;
}