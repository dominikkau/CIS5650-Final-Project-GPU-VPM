#pragma once

#include "common.h"

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

struct ParticleBuffer {
    const ParticleBufferType::Type bufferType;
    int bufferFields = 0;
    vpmvec3* X = NULL;          // Position
    vpmvec3* Gamma = NULL;      // Vectorial circulation
    vpmfloat* sigma = NULL;     // Smoothing radius
    int* index = NULL;          // Indices of particles
    vpmvec3* U = NULL;          // Velocity at particle
    vpmmat3* J = NULL;          // Jacobian at particle
    vpmmat3* M = NULL;          // Auxiliary memory
    vpmvec3* C = NULL;          // SFS coefficient, numerator, denominator
    vpmvec3* SFS = NULL;

    /*vpmfloat* vol = NULL;           // Volume
    vpmfloat* circulation = NULL;   // Scalar circulation
    bool* isStatic = NULL;          // Indicates if particle is static
    vpmvec3* PSE = NULL;            // Particle-strength exchange*/

    ParticleBuffer(ParticleBufferType::Type bufferType) : bufferType(bufferType) {};

    void mallocFields(unsigned int numParticles, int bufferMask);
    void freeFields();
    void freeFields(int bufferMask);
};

void _cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer,
    unsigned int destIndex, unsigned int srcNumParticles, int bufferMask, cudaStream_t stream = 0);
unsigned int cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer, unsigned int destNumParticles,
    unsigned int destMaxParticles, unsigned int srcNumParticles, unsigned int destIndex, int bufferMask, cudaStream_t stream = 0);