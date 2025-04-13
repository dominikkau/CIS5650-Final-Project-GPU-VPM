#include <cuda.h>
#include <cuda_runtime.h>
#include "particlebuffer.h"

// Allocates memory for fields in bufferMask
// Ignores fields that have been allocated already
void ParticleBuffer::mallocFields(unsigned int numParticles, int bufferMask) {
    // Ignore fields that have been allocated
    bufferMask &= ~bufferFields;

    switch (bufferType) {
    case BUFFER_DEVICE:
        if (bufferMask & BUFFER_X) {
            cudaMalloc((void**)&X, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of X failed!");
        }

        if (bufferMask & BUFFER_U) {
            cudaMalloc((void**)&U, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of U failed!");
        }

        if (bufferMask & BUFFER_J) {
            cudaMalloc((void**)&J, numParticles * sizeof(vpmmat3));
            checkCUDAError("cudaMalloc of J failed!");
        }

        if (bufferMask & BUFFER_GAMMA) {
            cudaMalloc((void**)&Gamma, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of Gamma failed!");
        }

        if (bufferMask & BUFFER_SIGMA) {
            cudaMalloc((void**)&sigma, numParticles * sizeof(vpmfloat));
            checkCUDAError("cudaMalloc of sigma failed!");
        }

        if (bufferMask & BUFFER_SFS) {
            cudaMalloc((void**)&SFS, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of SFS failed!");
        }

        if (bufferMask & BUFFER_C) {
            cudaMalloc((void**)&C, numParticles * sizeof(vpmvec3));
            checkCUDAError("cudaMalloc of C failed!");
        }

        if (bufferMask & BUFFER_M) {
            cudaMalloc((void**)&M, numParticles * sizeof(vpmmat3));
            checkCUDAError("cudaMalloc of M failed!");
        }

        if (bufferMask & BUFFER_INDEX) {
            cudaMalloc((void**)&index, numParticles * sizeof(int));
            checkCUDAError("cudaMalloc of index failed!");
        }

        /*cudaMalloc((void**)&PSE, size * sizeof(vpmvec3));
        checkCUDAError("cudaMalloc of PSE failed!");

        cudaMalloc((void**)&isStatic, size * sizeof(bool));
        checkCUDAError("cudaMalloc of isStatic failed!");

        cudaMalloc((void**)&vol, size * sizeof(vpmfloat));
        checkCUDAError("cudaMalloc of vol failed!");

        cudaMalloc((void**)&circulation, size * sizeof(vpmfloat));
        checkCUDAError("cudaMalloc of circulation failed!");*/
        break;

    case BUFFER_HOST:
        if (bufferMask & BUFFER_X) X = new vpmvec3[numParticles];
        if (bufferMask & BUFFER_U) U = new vpmvec3[numParticles];
        if (bufferMask & BUFFER_J) J = new vpmmat3[numParticles];
        if (bufferMask & BUFFER_GAMMA) Gamma = new vpmvec3[numParticles];
        if (bufferMask & BUFFER_SIGMA) sigma = new vpmfloat[numParticles];
        if (bufferMask & BUFFER_SFS) SFS = new vpmvec3[numParticles];
        if (bufferMask & BUFFER_C) C = new vpmvec3[numParticles];
        if (bufferMask & BUFFER_M) M = new vpmmat3[numParticles];
        if (bufferMask & BUFFER_INDEX) index = new int[numParticles];
        /*if (bufferMask & BUFFER_PSE) PSE = new vpmvec3[numParticles];
        if (bufferMask & BUFFER_IS_STATIC) isStatic = new bool[numParticles];
        if (bufferMask & BUFFER_VOL) vol = new vpmfloat[numParticles];
        if (bufferMask & BUFFER_CIRC) circulation = new vpmfloat[numParticles];*/
        break;

    case BUFFER_HOST_PINNED:
        if (bufferMask & BUFFER_X) cudaMallocHost((void**)&X, numParticles * sizeof(vpmvec3));
        if (bufferMask & BUFFER_U) cudaMallocHost((void**)&U, numParticles * sizeof(vpmvec3));
        if (bufferMask & BUFFER_J) cudaMallocHost((void**)&J, numParticles * sizeof(vpmmat3));
        if (bufferMask & BUFFER_GAMMA) cudaMallocHost((void**)&Gamma, numParticles * sizeof(vpmvec3));
        if (bufferMask & BUFFER_SIGMA) cudaMallocHost((void**)&sigma, numParticles * sizeof(vpmfloat));
        if (bufferMask & BUFFER_SFS) cudaMallocHost((void**)&SFS, numParticles * sizeof(vpmvec3));
        if (bufferMask & BUFFER_C) cudaMallocHost((void**)&C, numParticles * sizeof(vpmvec3));
        if (bufferMask & BUFFER_M) cudaMallocHost((void**)&M, numParticles * sizeof(vpmmat3));
        if (bufferMask & BUFFER_INDEX) cudaMallocHost((void**)&index, numParticles * sizeof(int));
        /*if (bufferMask & BUFFER_PSE) cudaMallocHost((void**)&PSE, numParticles * sizeof(vpmvec3));
        if (bufferMask & BUFFER_IS_STATIC) cudaMallocHost((void**)&isStatic, numParticles * sizeof(bool));
        if (bufferMask & BUFFER_VOL) cudaMallocHost((void**)&vol, numParticles * sizeof(vpmfloat));
        if (bufferMask & BUFFER_CIRC) cudaMallocHost((void**)&circulation, numParticles * sizeof(vpmfloat));*/
        break;
    }

    // Update allocated fields
    bufferFields |= bufferMask;
}

// Frees memory for fields in bufferMask
// Ignores fields that have not been allocated
void ParticleBuffer::freeFields(int bufferMask) {
    // Ignore unallocated fields
    bufferMask &= bufferFields;

    switch (bufferType) {
    case BUFFER_DEVICE:
        if (bufferMask & BUFFER_X) cudaFree(X);
        if (bufferMask & BUFFER_U) cudaFree(U);
        if (bufferMask & BUFFER_J) cudaFree(J);
        if (bufferMask & BUFFER_GAMMA) cudaFree(Gamma);
        if (bufferMask & BUFFER_SIGMA) cudaFree(sigma);
        if (bufferMask & BUFFER_SFS) cudaFree(SFS);
        if (bufferMask & BUFFER_C) cudaFree(C);
        if (bufferMask & BUFFER_M) cudaFree(M);
        if (bufferMask & BUFFER_INDEX) cudaFree(index);
        /*if (bufferMask & BUFFER_PSE) cudaFree(PSE);
        if (bufferMask & BUFFER_IS_STATIC) cudaFree(isStatic);
        if (bufferMask & BUFFER_VOL) cudaFree(vol);
        if (bufferMask & BUFFER_CIRC) cudaFree(circulation);*/
        break;

    case BUFFER_HOST:
        if (bufferMask & BUFFER_X) delete[] X;
        if (bufferMask & BUFFER_U) delete[] U;
        if (bufferMask & BUFFER_J) delete[] J;
        if (bufferMask & BUFFER_GAMMA) delete[] Gamma;
        if (bufferMask & BUFFER_SIGMA) delete[] sigma;
        if (bufferMask & BUFFER_SFS) delete[] SFS;
        if (bufferMask & BUFFER_C) delete[] C;
        if (bufferMask & BUFFER_M) delete[] M;
        if (bufferMask & BUFFER_INDEX) delete[] index;
        /*if (bufferMask & BUFFER_PSE) delete[] PSE;
        if (bufferMask & BUFFER_IS_STATIC) delete[] isStatic;
        if (bufferMask & BUFFER_VOL) delete[] vol;
        if (bufferMask & BUFFER_CIRC) delete[] circulation;*/
        break;

    case BUFFER_HOST_PINNED:
        if (bufferMask & BUFFER_X) cudaFreeHost(X);
        if (bufferMask & BUFFER_U) cudaFreeHost(U);
        if (bufferMask & BUFFER_J) cudaFreeHost(J);
        if (bufferMask & BUFFER_GAMMA) cudaFreeHost(Gamma);
        if (bufferMask & BUFFER_SIGMA) cudaFreeHost(sigma);
        if (bufferMask & BUFFER_SFS) cudaFreeHost(SFS);
        if (bufferMask & BUFFER_C) cudaFreeHost(C);
        if (bufferMask & BUFFER_M) cudaFreeHost(M);
        if (bufferMask & BUFFER_INDEX) cudaFreeHost(index);
        /*if (bufferMask & BUFFER_PSE) cudaFreeHost(PSE);
        if (bufferMask & BUFFER_IS_STATIC) cudaFreeHost(isStatic);
        if (bufferMask & BUFFER_VOL) cudaFreeHost(vol);
        if (bufferMask & BUFFER_CIRC) cudaFreeHost(circulation);*/
        break;
    }

    // Update allocated fields
    bufferFields &= ~bufferMask;
}

void _cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer,
    unsigned int destIndex, unsigned int srcNumParticles, int bufferMask, cudaStream_t stream) {

    // Determine cudaMemcpy direction
    cudaMemcpyKind cpyDirection;
    if (destBuffer.bufferType == BUFFER_DEVICE) {
        if (srcBuffer.bufferType == BUFFER_DEVICE) cpyDirection = cudaMemcpyDeviceToDevice;
        else cpyDirection = cudaMemcpyHostToDevice;
    }
    else {
        if (srcBuffer.bufferType == BUFFER_DEVICE) cpyDirection = cudaMemcpyDeviceToHost;
        else cpyDirection = cudaMemcpyHostToHost;
    }

    // Ensure that we do not try to copy from or to non-existent fields
    bufferMask &= (destBuffer.bufferFields & srcBuffer.bufferFields);

    if (bufferMask & BUFFER_X) {
        cudaMemcpyAsync(destBuffer.X + destIndex, srcBuffer.X, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_U) {
        cudaMemcpyAsync(destBuffer.U + destIndex, srcBuffer.U, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_J) {
        cudaMemcpyAsync(destBuffer.J + destIndex, srcBuffer.J, srcNumParticles * sizeof(vpmmat3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_GAMMA) {
        cudaMemcpyAsync(destBuffer.Gamma + destIndex, srcBuffer.Gamma, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_SIGMA) {
        cudaMemcpyAsync(destBuffer.sigma + destIndex, srcBuffer.sigma, srcNumParticles * sizeof(vpmfloat), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_SFS) {
        cudaMemcpyAsync(destBuffer.SFS + destIndex, srcBuffer.SFS, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_C) {
        cudaMemcpyAsync(destBuffer.C + destIndex, srcBuffer.C, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_M) {
        cudaMemcpyAsync(destBuffer.M + destIndex, srcBuffer.M, srcNumParticles * sizeof(vpmmat3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_INDEX) {
        cudaMemcpyAsync(destBuffer.index + destIndex, srcBuffer.index, srcNumParticles * sizeof(int), cpyDirection, stream);
    }
    /*if (bufferMask & BUFFER_PSE) {
        cudaMemcpyAsync(destBuffer.PSE + destIndex, srcBuffer.PSE, srcNumParticles * sizeof(vpmvec3), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_IS_STATIC) {
        cudaMemcpyAsync(destBuffer.isStatic + destIndex, srcBuffer.isStatic, srcNumParticles * sizeof(bool), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_VOL) {
        cudaMemcpyAsync(destBuffer.vol + destIndex, srcBuffer.vol, srcNumParticles * sizeof(vpmfloat), cpyDirection, stream);
    }
    if (bufferMask & BUFFER_CIRC) {
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