#include <cuda.h>
#include <cuda_runtime.h>
#include "particlebuffer.h"

// Allocates memory for fields in bufferMask
// Ignores fields that have been allocated already
void ParticleBuffer::mallocFields(unsigned int numParticles, int bufferMask) {
    // Ignore fields that have been allocated
    bufferMask &= ~bufferFields;

    switch (bufferType) {
    case ParticleBufferType::DEVICE:
        if (bufferMask & BufferField::X) {
            cudaMalloc((void**)&X, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of X failed!");
        }

        if (bufferMask & BufferField::U) {
            cudaMalloc((void**)&U, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of U failed!");
        }

        if (bufferMask & BufferField::J) {
            cudaMalloc((void**)&J, numParticles * sizeof(vpmmat3));
            checkCUDAError("cudaMalloc of J failed!");
        }

        if (bufferMask & BufferField::GAMMA) {
            cudaMalloc((void**)&Gamma, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of Gamma failed!");
        }

        if (bufferMask & BufferField::SIGMA) {
            cudaMalloc((void**)&sigma, numParticles * sizeof(vpmfloat));
            checkCUDAError("cudaMalloc of sigma failed!");
        }

        if (bufferMask & BufferField::SFS) {
            cudaMalloc((void**)&SFS, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of SFS failed!");
        }

        if (bufferMask & BufferField::C) {
            cudaMalloc((void**)&C, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of C failed!");
        }

        if (bufferMask & BufferField::M) {
            cudaMalloc((void**)&M, numParticles * sizeof(vpmmat3));
            checkCUDAError("cudaMalloc of M failed!");
        }

        if (bufferMask & BufferField::INDEX) {
            cudaMalloc((void**)&index, numParticles * sizeof(int));
            checkCUDAError("cudaMalloc of index failed!");
        }

        /*if (bufferMask & BufferField::PSE) {
            cudaMalloc((void**)&PSE, size * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of PSE failed!");
        }

        if (bufferMask & BufferField::IS_STATIC) {
            cudaMalloc((void**)&isStatic, size * sizeof(bool));
            checkCUDAError("cudaMalloc of isStatic failed!");
        }

        if (bufferMask & BufferField::VOL) {
            cudaMalloc((void**)&vol, size * sizeof(vpmfloat));
            checkCUDAError("cudaMalloc of vol failed!");
        }

        if (bufferMask & BufferField::CIRC) {
            cudaMalloc((void**)&circulation, size * sizeof(vpmfloat));
            checkCUDAError("cudaMalloc of circulation failed!");
        }*/
        break;

    case ParticleBufferType::HOST:
        if (bufferMask & BufferField::X) X = new vpmvec3[numParticles];
        if (bufferMask & BufferField::U) U = new vpmvec3[numParticles];
        if (bufferMask & BufferField::J) J = new vpmmat3[numParticles];
        if (bufferMask & BufferField::GAMMA) Gamma = new vpmvec3[numParticles];
        if (bufferMask & BufferField::SIGMA) sigma = new vpmfloat[numParticles];
        if (bufferMask & BufferField::SFS) SFS = new vpmvec3[numParticles];
        if (bufferMask & BufferField::C) C = new vpmvec3[numParticles];
        if (bufferMask & BufferField::M) M = new vpmmat3[numParticles];
        if (bufferMask & BufferField::INDEX) index = new int[numParticles];
        /*if (bufferMask & BufferField::PSE) PSE = new vpmvec3[numParticles];
        if (bufferMask & BufferField::IS_STATIC) isStatic = new bool[numParticles];
        if (bufferMask & BufferField::VOL) vol = new vpmfloat[numParticles];
        if (bufferMask & BufferField::CIRC) circulation = new vpmfloat[numParticles];*/
        break;

    case ParticleBufferType::HOST_PINNED:
        if (bufferMask & BufferField::X) cudaMallocHost((void**)&X, numParticles * sizeof(vpmvec3));
        if (bufferMask & BufferField::U) cudaMallocHost((void**)&U, numParticles * sizeof(vpmvec3));
        if (bufferMask & BufferField::J) cudaMallocHost((void**)&J, numParticles * sizeof(vpmmat3));
        if (bufferMask & BufferField::GAMMA) cudaMallocHost((void**)&Gamma, numParticles * sizeof(vpmvec3));
        if (bufferMask & BufferField::SIGMA) cudaMallocHost((void**)&sigma, numParticles * sizeof(vpmfloat));
        if (bufferMask & BufferField::SFS) cudaMallocHost((void**)&SFS, numParticles * sizeof(vpmvec3));
        if (bufferMask & BufferField::C) cudaMallocHost((void**)&C, numParticles * sizeof(vpmvec3));
        if (bufferMask & BufferField::M) cudaMallocHost((void**)&M, numParticles * sizeof(vpmmat3));
        if (bufferMask & BufferField::INDEX) cudaMallocHost((void**)&index, numParticles * sizeof(int));
        /*if (bufferMask & BufferField::PSE) cudaMallocHost((void**)&PSE, numParticles * sizeof(vpmvec3));
        if (bufferMask & BufferField::IS_STATIC) cudaMallocHost((void**)&isStatic, numParticles * sizeof(bool));
        if (bufferMask & BufferField::VOL) cudaMallocHost((void**)&vol, numParticles * sizeof(vpmfloat));
        if (bufferMask & BufferField::CIRC) cudaMallocHost((void**)&circulation, numParticles * sizeof(vpmfloat));*/
        break;
    }

    // Update allocated fields
    bufferFields |= bufferMask;
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
        if (bufferMask & BufferField::X) cudaFree(X);
        if (bufferMask & BufferField::U) cudaFree(U);
        if (bufferMask & BufferField::J) cudaFree(J);
        if (bufferMask & BufferField::GAMMA) cudaFree(Gamma);
        if (bufferMask & BufferField::SIGMA) cudaFree(sigma);
        if (bufferMask & BufferField::SFS) cudaFree(SFS);
        if (bufferMask & BufferField::C) cudaFree(C);
        if (bufferMask & BufferField::M) cudaFree(M);
        if (bufferMask & BufferField::INDEX) cudaFree(index);
        /*if (bufferMask & BufferField::PSE) cudaFree(PSE);
        if (bufferMask & BufferField::IS_STATIC) cudaFree(isStatic);
        if (bufferMask & BufferField::VOL) cudaFree(vol);
        if (bufferMask & BufferField::CIRC) cudaFree(circulation);*/
        break;

    case ParticleBufferType::HOST:
        if (bufferMask & BufferField::X) delete[] X;
        if (bufferMask & BufferField::U) delete[] U;
        if (bufferMask & BufferField::J) delete[] J;
        if (bufferMask & BufferField::GAMMA) delete[] Gamma;
        if (bufferMask & BufferField::SIGMA) delete[] sigma;
        if (bufferMask & BufferField::SFS) delete[] SFS;
        if (bufferMask & BufferField::C) delete[] C;
        if (bufferMask & BufferField::M) delete[] M;
        if (bufferMask & BufferField::INDEX) delete[] index;
        /*if (bufferMask & BufferField::PSE) delete[] PSE;
        if (bufferMask & BufferField::IS_STATIC) delete[] isStatic;
        if (bufferMask & BufferField::VOL) delete[] vol;
        if (bufferMask & BufferField::CIRC) delete[] circulation;*/
        break;

    case ParticleBufferType::HOST_PINNED:
        if (bufferMask & BufferField::X) cudaFreeHost(X);
        if (bufferMask & BufferField::U) cudaFreeHost(U);
        if (bufferMask & BufferField::J) cudaFreeHost(J);
        if (bufferMask & BufferField::GAMMA) cudaFreeHost(Gamma);
        if (bufferMask & BufferField::SIGMA) cudaFreeHost(sigma);
        if (bufferMask & BufferField::SFS) cudaFreeHost(SFS);
        if (bufferMask & BufferField::C) cudaFreeHost(C);
        if (bufferMask & BufferField::M) cudaFreeHost(M);
        if (bufferMask & BufferField::INDEX) cudaFreeHost(index);
        /*if (bufferMask & BufferField::PSE) cudaFreeHost(PSE);
        if (bufferMask & BufferField::IS_STATIC) cudaFreeHost(isStatic);
        if (bufferMask & BufferField::VOL) cudaFreeHost(vol);
        if (bufferMask & BufferField::CIRC) cudaFreeHost(circulation);*/
        break;
    }

    // Update allocated fields
    bufferFields &= ~bufferMask;
}

void _cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer,
    unsigned int destIndex, unsigned int srcNumParticles, int bufferMask, cudaStream_t stream) {

    // Determine cudaMemcpy direction
    cudaMemcpyKind cpyDirection;
    if (destBuffer.bufferType == ParticleBufferType::DEVICE) {
        if (srcBuffer.bufferType == ParticleBufferType::DEVICE) cpyDirection = cudaMemcpyDeviceToDevice;
        else cpyDirection = cudaMemcpyHostToDevice;
    }
    else {
        if (srcBuffer.bufferType == ParticleBufferType::DEVICE) cpyDirection = cudaMemcpyDeviceToHost;
        else cpyDirection = cudaMemcpyHostToHost;
    }

    // Ensure that we do not try to copy from or to non-existent fields
    bufferMask &= (destBuffer.bufferFields & srcBuffer.bufferFields);

    if (bufferMask & BufferField::X) {
        cudaMemcpyAsync(destBuffer.X + destIndex, srcBuffer.X, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::U) {
        cudaMemcpyAsync(destBuffer.U + destIndex, srcBuffer.U, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::J) {
        cudaMemcpyAsync(destBuffer.J + destIndex, srcBuffer.J, srcNumParticles * sizeof(vpmmat3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::GAMMA) {
        cudaMemcpyAsync(destBuffer.Gamma + destIndex, srcBuffer.Gamma, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::SIGMA) {
        cudaMemcpyAsync(destBuffer.sigma + destIndex, srcBuffer.sigma, srcNumParticles * sizeof(vpmfloat), cpyDirection, stream);
    }
    if (bufferMask & BufferField::SFS) {
        cudaMemcpyAsync(destBuffer.SFS + destIndex, srcBuffer.SFS, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::C) {
        cudaMemcpyAsync(destBuffer.C + destIndex, srcBuffer.C, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::M) {
        cudaMemcpyAsync(destBuffer.M + destIndex, srcBuffer.M, srcNumParticles * sizeof(vpmmat3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::INDEX) {
        cudaMemcpyAsync(destBuffer.index + destIndex, srcBuffer.index, srcNumParticles * sizeof(int), cpyDirection, stream);
    }
    /*if (bufferMask & BufferField::PSE) {
        cudaMemcpyAsync(destBuffer.PSE + destIndex, srcBuffer.PSE, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BufferField::IS_STATIC) {
        cudaMemcpyAsync(destBuffer.isStatic + destIndex, srcBuffer.isStatic, srcNumParticles * sizeof(bool), cpyDirection, stream);
    }
    if (bufferMask & BufferField::VOL) {
        cudaMemcpyAsync(destBuffer.vol + destIndex, srcBuffer.vol, srcNumParticles * sizeof(vpmfloat), cpyDirection, stream);
    }
    if (bufferMask & BufferField::CIRC) {
        cudaMemcpyAsync(destBuffer.circulation + destIndex, srcBuffer.circulation, srcNumParticles * sizeof(vpmfloat), cpyDirection, stream);
    }*/
}

unsigned int cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer, unsigned int destNumParticles,
    unsigned int destMaxParticles, unsigned int srcNumParticles, unsigned int destIndex, int bufferMask, cudaStream_t stream) {

    // Start index exceeds maximum number of particles
    if (destIndex >= destMaxParticles) return destNumParticles;

    // Do not leave undefined particles between existing and copied
    if (destIndex > destNumParticles) destIndex = destNumParticles;

    // Number of particles to be copied is limited by destMaxParticles
    srcNumParticles = min(srcNumParticles, destMaxParticles - destIndex);

    _cpyParticleBuffer(destBuffer, srcBuffer, destIndex, srcNumParticles, bufferMask, stream);

    // Calculate new number of particles
    if (destIndex + srcNumParticles >= destNumParticles) {
        destNumParticles = destIndex + srcNumParticles;
    }

    return destNumParticles;
}